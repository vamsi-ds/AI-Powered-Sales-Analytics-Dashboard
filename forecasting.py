"""
Forecasting Module - Machine Learning models for sales prediction and trend forecasting.
Implements Prophet and XGBoost for time series forecasting.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


class SalesForecaster:
    """
    Sales forecasting using multiple ML approaches.
    Supports Prophet, XGBoost, and ensemble methods.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = {}
        self.results = {}

        # Ensure Order_Date is datetime
        if 'Order_Date' in self.df.columns:
            self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'], errors='coerce')

    def prepare_timeseries_data(self, freq: str = 'D') -> pd.DataFrame:
        """
        Prepare time series data for forecasting.
        Aggregates sales data to daily, weekly, or monthly frequency.
        """
        self.logger.info(f"Preparing time series data (freq={freq})...")

        date_col = 'Order_Date'

        if freq == 'D':
            ts_data = self.df.groupby(date_col).agg(
                Sales=('Sales', 'sum'),
                Profit=('Profit', 'sum'),
                Orders=('Order_ID', 'nunique'),
                Quantity=('Quantity', 'sum'),
                Avg_Discount=('Discount', 'mean')
            ).reset_index().sort_values(date_col)
        elif freq == 'W':
            ts_data = self.df.groupby(pd.Grouper(key=date_col, freq='W')).agg(
                Sales=('Sales', 'sum'),
                Profit=('Profit', 'sum'),
                Orders=('Order_ID', 'nunique'),
                Quantity=('Quantity', 'sum'),
                Avg_Discount=('Discount', 'mean')
            ).reset_index()
        elif freq == 'M':
            ts_data = self.df.groupby(pd.Grouper(key=date_col, freq='M')).agg(
                Sales=('Sales', 'sum'),
                Profit=('Profit', 'sum'),
                Orders=('Order_ID', 'nunique'),
                Quantity=('Quantity', 'sum'),
                Avg_Discount=('Discount', 'mean')
            ).reset_index()
        elif freq == 'Q':
            ts_data = self.df.groupby(pd.Grouper(key=date_col, freq='Q')).agg(
                Sales=('Sales', 'sum'),
                Profit=('Profit', 'sum'),
                Orders=('Order_ID', 'nunique'),
                Quantity=('Quantity', 'sum'),
                Avg_Discount=('Discount', 'mean')
            ).reset_index()
        else:
            raise ValueError(f"Unsupported frequency: {freq}")

        # Handle missing dates by forward filling
        ts_data = ts_data.ffill().fillna(0)
        self.logger.info(f"  Created {len(ts_data):,} {freq} data points")
        return ts_data

    def forecast_prophet(self, ts_data: pd.DataFrame, periods: int = 30,
                         freq: str = 'D') -> dict:
        """
        Forecast using Facebook Prophet.
        Prophet is robust to missing data and handles seasonality well.
        """
        self.logger.info("=" * 60)
        self.logger.info("PROPHET FORECASTING")
        self.logger.info("=" * 60)

        try:
            from prophet import Prophet
            from prophet.diagnostics import cross_validation, performance_metrics
        except ImportError:
            self.logger.error(
                "Prophet not installed. Run: pip install prophet"
            )
            return self._fallback_forecast(ts_data, periods, freq)

        # Prepare data for Prophet
        prophet_df = ts_data[['Order_Date', 'Sales']].copy()
        prophet_df.columns = ['ds', 'y']

        # Ensure no zero or negative values for Prophet
        prophet_df['y'] = prophet_df['y'].clip(lower=0)

        if len(prophet_df) < 10:
            self.logger.warning("Too few data points for reliable Prophet forecasting")
            return self._fallback_forecast(ts_data, periods, freq)

        try:
            # Create and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=(freq == 'D'),
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                interval_width=0.95
            )

            # Add monthly seasonality for daily data
            if freq == 'D':
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            model.fit(prophet_df)

            # Make future dataframe
            future = model.make_future_dataframe(periods=periods, freq=freq)
            forecast = model.predict(future)

            # Extract components
            actual = prophet_df
            predicted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            predicted.columns = ['Order_Date', 'Sales_Forecast', 'Lower_Bound', 'Upper_Bound']

            # Merge actual with forecast
            result_df = predicted.merge(
                actual.rename(columns={'ds': 'Order_Date', 'y': 'Actual_Sales'}),
                on='Order_Date',
                how='left'
            )

            # Get forecast only (future periods)
            future_forecast = result_df[result_df['Actual_Sales'].isna()].copy()

            # Model performance
            train_score = self._calculate_forecast_accuracy(
                result_df.dropna(subset=['Actual_Sales']),
                'Actual_Sales', 'Sales_Forecast'
            )

            # Trend and seasonality components
            trend = forecast[['ds', 'trend']].copy()
            trend.columns = ['Order_Date', 'Trend']

            yearly_seas = forecast[['ds', 'yearly']].copy()
            yearly_seas.columns = ['Order_Date', 'Yearly_Seasonality']

            self.logger.info(f"  MAPE: {train_score['mape']:.2f}%")
            self.logger.info(f"  RMSE: ${train_score['rmse']:.2f}")

            result = {
                'model': model,
                'forecast': result_df,
                'future_forecast': future_forecast,
                'trend': trend,
                'yearly_seasonality': yearly_seas,
                'accuracy': train_score,
                'model_type': 'prophet'
            }

            # Save model
            self._save_model_artifacts(result, 'prophet')

            return result

        except Exception as e:
            self.logger.error(f"Prophet forecasting failed: {e}")
            return self._fallback_forecast(ts_data, periods, freq)

    def forecast_xgboost(self, ts_data: pd.DataFrame, periods: int = 30,
                         freq: str = 'D') -> dict:
        """
        Forecast using XGBoost with feature engineering.
        Captures non-linear patterns and complex interactions.
        """
        self.logger.info("=" * 60)
        self.logger.info("XGBOOST FORECASTING")
        self.logger.info("=" * 60)

        try:
            import xgboost as xgb
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        except ImportError:
            self.logger.error(
                "XGBoost not installed. Run: pip install xgboost"
            )
            return self._fallback_forecast(ts_data, periods, freq)

        df = ts_data.copy()

        if len(df) < 20:
            self.logger.warning("Too few data points for reliable XGBoost forecasting")
            return self._fallback_forecast(ts_data, periods, freq)

        try:
            # Create time-based features
            df['Year'] = df['Order_Date'].dt.year
            df['Month'] = df['Order_Date'].dt.month
            df['Day'] = df['Order_Date'].dt.day
            df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
            df['Quarter'] = df['Order_Date'].dt.quarter
            df['DayOfYear'] = df['Order_Date'].dt.dayofyear
            df['WeekOfYear'] = df['Order_Date'].dt.isocalendar().week.astype(int)
            df['IsMonthStart'] = (df['Day'] <= 7).astype(int)
            df['IsMonthEnd'] = (df['Day'] >= 21).astype(int)
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

            # Lag features
            df['Sales_Lag_1'] = df['Sales'].shift(1)
            df['Sales_Lag_2'] = df['Sales'].shift(2)
            df['Sales_Lag_3'] = df['Sales'].shift(3)
            df['Sales_Lag_7'] = df['Sales'].shift(7)

            # Rolling features
            df['Sales_Rolling_7'] = df['Sales'].rolling(window=7).mean()
            df['Sales_Rolling_14'] = df['Sales'].rolling(window=14).mean()
            df['Sales_Rolling_30'] = df['Sales'].rolling(window=30).mean()

            # Drop NaN rows from lag features
            df = df.dropna()

            if len(df) < 10:
                self.logger.warning("Not enough data after creating lag features")
                return self._fallback_forecast(ts_data, periods, freq)

            # Feature selection
            feature_cols = [
                'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'DayOfYear',
                'WeekOfYear', 'IsMonthStart', 'IsMonthEnd', 'IsWeekend',
                'Sales_Lag_1', 'Sales_Lag_2', 'Sales_Lag_3', 'Sales_Lag_7',
                'Sales_Rolling_7', 'Sales_Rolling_14', 'Sales_Rolling_30'
            ]

            X = df[feature_cols]
            y = df['Sales']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train/Test split (time-based)
            split_idx = int(len(X_scaled) * 0.8)
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Predictions
            y_pred = model.predict(X_test)

            # Model performance
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # For future forecasting, we need to iteratively predict
            future_dates = pd.date_range(
                start=df['Order_Date'].max() + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )

            future_predictions = []
            last_row = df.iloc[-1:].copy()

            for future_date in future_dates:
                future_features = pd.DataFrame([{
                    'Year': future_date.year,
                    'Month': future_date.month,
                    'Day': future_date.day,
                    'DayOfWeek': future_date.dayofweek,
                    'Quarter': future_date.quarter,
                    'DayOfYear': future_date.dayofyear,
                    'WeekOfYear': future_date.isocalendar().week,
                    'IsMonthStart': 1 if future_date.day <= 7 else 0,
                    'IsMonthEnd': 1 if future_date.day >= 21 else 0,
                    'IsWeekend': 1 if future_date.dayofweek >= 5 else 0,
                    'Sales_Lag_1': last_row['Sales'].values[0],
                    'Sales_Lag_2': last_row['Sales_Lag_1'].values[0],
                    'Sales_Lag_3': last_row['Sales_Lag_2'].values[0],
                    'Sales_Lag_7': last_row['Sales_Lag_7'].values[0],
                    'Sales_Rolling_7': last_row['Sales_Rolling_7'].values[0],
                    'Sales_Rolling_14': last_row['Sales_Rolling_14'].values[0],
                    'Sales_Rolling_30': last_row['Sales_Rolling_30'].values[0],
                }])

                future_features_scaled = scaler.transform(future_features[feature_cols])
                pred = model.predict(future_features_scaled)[0]

                future_predictions.append({
                    'Order_Date': future_date,
                    'Sales_Forecast': max(pred, 0),  # No negative sales
                    'Lower_Bound': max(pred * 0.8, 0),
                    'Upper_Bound': pred * 1.2
                })

                # Update last_row for next iteration
                new_row = last_row.copy()
                new_row['Sales'] = pred
                new_row['Sales_Lag_1'] = new_row['Sales'].values[0]
                new_row['Sales_Lag_2'] = new_row['Sales_Lag_1'].values[0]
                new_row['Sales_Lag_3'] = new_row['Sales_Lag_2'].values[0]
                new_row['Sales_Lag_7'] = new_row['Sales_Lag_7'].values[0]
                new_row['Sales_Rolling_7'] = new_row['Sales_Rolling_7'].values[0]
                new_row['Sales_Rolling_14'] = new_row['Sales_Rolling_14'].values[0]
                new_row['Sales_Rolling_30'] = new_row['Sales_Rolling_30'].values[0]
                last_row = new_row

            future_forecast = pd.DataFrame(future_predictions)

            # Combined results
            result_df = pd.concat([
                df[['Order_Date', 'Sales']].rename(columns={'Sales': 'Actual_Sales'}),
                future_forecast.rename(columns={'Sales_Forecast': 'Sales_Forecast'})[['Order_Date']]
            ], ignore_index=True)

            # Merge forecast with actual
            forecast_result = future_forecast.copy()

            self.logger.info(f"  MAE: ${mae:.2f}")
            self.logger.info(f"  RMSE: ${rmse:.2f}")
            self.logger.info(f"  R²: {r2:.3f}")
            self.logger.info(f"  MAPE: {mape:.2f}%")

            result = {
                'model': model,
                'forecast': result_df,
                'future_forecast': future_forecast,
                'feature_importance': feature_importance,
                'accuracy': {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape},
                'model_type': 'xgboost',
                'scaler': scaler
            }

            # Save model
            self._save_model_artifacts(result, 'xgboost')

            return result

        except Exception as e:
            self.logger.error(f"XGBoost forecasting failed: {e}")
            return self._fallback_forecast(ts_data, periods, freq)

    def _fallback_forecast(self, ts_data: pd.DataFrame, periods: int = 30,
                           freq: str = 'D') -> dict:
        """
        Fallback forecasting using simple statistical methods.
        Uses moving average and trend decomposition.
        """
        self.logger.info("Using fallback statistical forecasting...")

        df = ts_data.copy()

        # Simple moving average forecast
        if len(df) >= 7:
            last_values = df['Sales'].tail(7).values
            seasonal_pattern = df['Sales'].tail(90).values if len(df) >= 90 else last_values
        else:
            last_values = df['Sales'].values
            seasonal_pattern = last_values

        # Generate forecast
        last_date = df['Order_Date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq=freq
        )

        # Simple seasonal naive forecast
        base_forecast = np.tile(seasonal_pattern, int(np.ceil(periods / len(seasonal_pattern))))[:periods]

        # Add noise for realism
        noise = np.random.normal(0, base_forecast.std() * 0.1, periods)
        forecast_values = base_forecast + noise
        forecast_values = np.maximum(forecast_values, 0)  # No negative values

        future_forecast = pd.DataFrame({
            'Order_Date': future_dates,
            'Sales_Forecast': forecast_values,
            'Lower_Bound': forecast_values * 0.7,
            'Upper_Bound': forecast_values * 1.3
        })

        result = {
            'model': None,
            'forecast': None,
            'future_forecast': future_forecast,
            'accuracy': {'mape': 0, 'rmse': 0, 'method': 'statistical_fallback'},
            'model_type': 'statistical'
        }

        self.logger.info("  Fallback forecast generated using seasonal naive method")
        return result

    def _calculate_forecast_accuracy(self, df: pd.DataFrame,
                                     actual_col: str,
                                     forecast_col: str) -> dict:
        """Calculate forecast accuracy metrics."""
        actual = df[actual_col].values
        forecast = df[forecast_col].values

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - forecast) / (actual + 1e-10))) * 100

        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))

        # Mean Absolute Error
        mae = np.mean(np.abs(actual - forecast))

        # Symmetric MAPE
        smape = np.mean(
            2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast) + 1e-10)
        ) * 100

        return {
            'mape': float(mape),
            'rmse': float(rmse),
            'mae': float(mae),
            'smape': float(smape)
        }

    def _save_model_artifacts(self, result: dict, model_name: str):
        """Save model artifacts to disk."""

        model = result.get('model')
        if model is not None:
            try:
                path = MODELS_DIR / f'{model_name}_model.pkl'
                joblib.dump(model, path)
                self.logger.info(f"  Model saved to {path}")
            except Exception as e:
                self.logger.warning(f"  Could not save model: {e}")

        # Save future forecast as CSV
        forecast = result.get('future_forecast')
        if forecast is not None:
            path = MODELS_DIR / f'{model_name}_forecast.csv'
            forecast.to_csv(path, index=False)
            self.logger.info(f"  Forecast saved to {path}")

        # Save accuracy metrics
        accuracy = result.get('accuracy', {})
        if accuracy:
            path = MODELS_DIR / f'{model_name}_metrics.json'
            with open(path, 'w') as f:
                json.dump(accuracy, f, indent=2, default=str)
            self.logger.info(f"  Metrics saved to {path}")

    def detect_anomalies(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Detect anomalies in sales data using statistical methods.

        Methods:
        - iqr: Interquartile range based
        - zscore: Z-score based
        - isolation_forest: ML-based anomaly detection
        """
        self.logger.info("=" * 60)
        self.logger.info("ANOMALY DETECTION")
        self.logger.info("=" * 60)

        daily_sales = self.df.groupby('Order_Date').agg(
            Sales=('Sales', 'sum'),
            Orders=('Order_ID', 'nunique'),
            Avg_Value=('Sales', 'mean')
        ).reset_index()

        anomalies = pd.DataFrame()

        if method == 'iqr':
            Q1 = daily_sales['Sales'].quantile(0.25)
            Q3 = daily_sales['Sales'].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            anomalies = daily_sales[
                (daily_sales['Sales'] < lower) | (daily_sales['Sales'] > upper)
            ].copy()
            anomalies['Anomaly_Type'] = np.where(
                anomalies['Sales'] > upper, 'Peak (High Sales)', 'Dip (Low Sales)'
            )

        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(daily_sales['Sales']))
            anomalies = daily_sales[z_scores > 2].copy()
            anomalies['Anomaly_Type'] = np.where(
                anomalies['Sales'] > daily_sales['Sales'].mean(), 'Peak', 'Dip'
            )

        elif method == 'isolation_forest':
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(
                    contamination=0.05, random_state=42
                )
                features = daily_sales[['Sales', 'Orders', 'Avg_Value']]
                predictions = iso_forest.fit_predict(features)
                daily_sales['Anomaly_Score'] = iso_forest.score_samples(features)
                anomalies = daily_sales[predictions == -1].copy()
                anomalies['Anomaly_Type'] = 'Isolation Forest'
            except ImportError:
                self.logger.warning("scikit-learn not available for isolation forest")
                return self.detect_anomalies('iqr')

        self.logger.info(f"  Found {len(anomalies):,} anomalies using {method} method")
        return anomalies

    def generate_business_insights(self) -> list:
        """
        Generate AI-powered business insights from the data.
        Uses statistical analysis to derive actionable recommendations.
        """
        self.logger.info("=" * 60)
        self.logger.info("GENERATING BUSINESS INSIGHTS")
        self.logger.info("=" * 60)

        insights = []
        df = self.df

        # Sales trends
        monthly_sales = df.groupby('Year_Month')['Sales'].sum()
        if len(monthly_sales) >= 2:
            growth = ((monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2] * 100)
            if growth > 10:
                insights.append(f"📈 Strong monthly sales growth of {growth:.1f}%. "
                                "Consider increasing inventory to meet rising demand.")
            elif growth < -10:
                insights.append(f"📉 Sales declined by {abs(growth):.1f}% this month. "
                                "Consider promotional campaigns to boost sales.")
            else:
                insights.append(f"📊 Sales are stable with {growth:.1f}% monthly change. "
                                "Focus on maintaining current momentum.")

        # Profit analysis
        avg_margin = df['Profit_Margin_Pct'].mean()
        if avg_margin > 30:
            insights.append(f"💰 Strong profit margin of {avg_margin:.1f}%. "
                            "Explore opportunities to expand market share.")
        elif avg_margin < 10:
            insights.append(f"⚠️ Low profit margin of {avg_margin:.1f}%. "
                            "Review pricing strategy and cost structure.")
        else:
            insights.append(f"📊 Healthy profit margin of {avg_margin:.1f}%. "
                            "Continue monitoring cost efficiency.")

        # Customer insights
        avg_orders = df['Customer_Total_Orders'].mean()
        avg_spent = df['Customer_Total_Spent'].mean()
        insights.append(f"👥 Average customer places {avg_orders:.1f} orders totaling "
                        f"${avg_spent:,.2f}. Implement loyalty programs to increase retention.")

        # Product insights
        top_category = df.groupby('Category')['Sales'].sum().idxmax()
        top_pct = df.groupby('Category')['Sales'].sum().max() / df['Sales'].sum() * 100
        insights.append(f"🏆 '{top_category}' is the top-performing category at {top_pct:.1f}% of total sales. "
                        "Consider expanding this product line.")

        # Seasonal insights
        if 'Season' in df.columns:
            peak_season = df.groupby('Season')['Sales'].sum().idxmax()
            insights.append(f"🌤️ '{peak_season}' season shows highest sales. "
                            "Plan marketing campaigns accordingly.")

        # Regional insights
        top_region = df.groupby('Region')['Sales'].sum().idxmax()
        insights.append(f"🌍 '{top_region}' region leads in sales. "
                        "Analyze success factors for replication in other regions.")

        # Payment insights
        top_payment = df.groupby('Payment_Mode')['Sales'].sum().idxmax()
        insights.append(f"💳 '{top_payment}' is the most popular payment method. "
                        "Ensure seamless payment experience.")

        # Discount impact
        high_discount = df[df['Discount'] > df['Discount'].quantile(0.75)]
        high_discount_margin = high_discount['Profit_Margin_Pct'].mean()
        low_discount = df[df['Discount'] <= df['Discount'].quantile(0.25)]
        low_discount_margin = low_discount['Profit_Margin_Pct'].mean()
        if high_discount_margin < low_discount_margin:
            insights.append(f"🏷️ High-discount orders show {abs(high_discount_margin - low_discount_margin):.1f}% "
                            "lower margins. Optimize discount strategy to protect profitability.")

        # Recommendations
        insights.append("💡 RECOMMENDATIONS:")
        recommendations = [
            "Focus on high-margin product categories to maximize profitability.",
            "Implement targeted marketing campaigns for top customer segments.",
            "Optimize inventory based on seasonal demand patterns.",
            "Develop customer retention programs to increase repeat purchases.",
            "Expand successful regional strategies to underperforming markets."
        ]
        insights.extend([f"  • {rec}" for rec in recommendations])

        self.logger.info(f"  Generated {len(insights)} insights")
        return insights

    def run_all_forecasts(self) -> dict:
        """Run all forecasting models and return results."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING ALL FORECASTING MODELS")
        self.logger.info("=" * 60)

        results = {}

        # Prepare monthly data for forecast
        monthly_data = self.prepare_timeseries_data('M')

        # Try Prophet first
        prophet_result = self.forecast_prophet(monthly_data, periods=6, freq='M')
        results['prophet'] = prophet_result

        # Try XGBoost
        xgb_result = self.forecast_xgboost(monthly_data, periods=6, freq='M')
        results['xgboost'] = xgb_result

        # Anomaly detection
        results['anomalies'] = self.detect_anomalies('iqr')

        # Business insights
        results['insights'] = self.generate_business_insights()

        self.logger.info("\nForecasting complete!")
        return results


if __name__ == '__main__':
    from data_loader import load_dataset
    from preprocessing import run_preprocessing_pipeline

    df = load_dataset()
    df = run_preprocessing_pipeline(df)
    forecaster = SalesForecaster(df)
    results = forecaster.run_all_forecasts()

    print("\n=== BUSINESS INSIGHTS ===")
    for insight in results['insights']:
        print(f"  {insight}")

    print(f"\n=== ANOMALIES FOUND: {len(results['anomalies'])} ===")
    if not results['anomalies'].empty:
        print(results['anomalies'].head(10).to_string())
