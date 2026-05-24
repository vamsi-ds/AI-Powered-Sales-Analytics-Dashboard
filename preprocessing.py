"""
Preprocessing Module - Handles data cleaning, validation, and feature engineering.
Ensures data quality and prepares features for analysis and modeling.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning pipeline.
    Handles missing values, duplicates, outliers, and data type conversion.
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    df = df.copy()

    # Log initial state
    initial_rows = len(df)
    logger.info(f"Initial rows: {initial_rows:,}")
    logger.info(f"Initial columns: {list(df.columns)}")

    # 1. Handle missing values
    logger.info("\n[1/6] Handling missing values...")
    missing_before = df.isnull().sum().sum()
    logger.info(f"  Missing values before: {missing_before}")

    df = _handle_missing_values(df)

    missing_after = df.isnull().sum().sum()
    logger.info(f"  Missing values after: {missing_after}")

    # 2. Remove duplicates
    logger.info("\n[2/6] Removing duplicates...")
    dup_count = df.duplicated(subset=['Order_ID']).sum() if 'Order_ID' in df.columns else 0
    logger.info(f"  Duplicate orders found: {dup_count}")
    if 'Order_ID' in df.columns:
        df = df.drop_duplicates(subset=['Order_ID'])
    else:
        df = df.drop_duplicates()

    # 3. Handle outliers in numeric columns
    logger.info("\n[3/6] Handling outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_cols = ['Sales', 'Profit', 'Discount', 'Quantity']
    for col in outlier_cols:
        if col in df.columns:
            outliers = _handle_outliers(df, col)
            if outliers > 0:
                logger.info(f"  {col}: capped {outliers} outliers")

    # 4. Ensure correct data types
    logger.info("\n[4/6] Converting data types...")
    df = _convert_dtypes(df)

    # 5. Remove negative or zero sales (likely data errors)
    if 'Sales' in df.columns:
        neg_sales = (df['Sales'] <= 0).sum()
        if neg_sales > 0:
            df = df[df['Sales'] > 0]
            logger.info(f"  Removed {neg_sales} records with non-positive sales")

    # 6. Validate required columns exist
    logger.info("\n[5/6] Validating required columns...")
    required = ['Order_ID', 'Customer_Name', 'Sales', 'Profit', 'Order_Date', 'Category']
    for col in required:
        if col not in df.columns:
            logger.warning(f"  Missing recommended column: {col}")

    # Log final state
    final_rows = len(df)
    logger.info(f"\n[6/6] Cleaning complete:")
    logger.info(f"  Rows removed: {initial_rows - final_rows:,}")
    logger.info(f"  Final rows: {final_rows:,}")
    logger.info(f"  Final columns: {list(df.columns)}")

    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in different column types."""
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if df[col].dtype in ['int64', 'float64']:
            # For numeric columns, fill with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
        elif df[col].dtype == 'object':
            # For categorical columns, fill with mode or 'Unknown'
            if missing_count < len(df) * 0.1:  # Less than 10% missing
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
                logger.info(f"  Filled {missing_count} missing values in '{col}' with mode ({mode_val})")
            else:
                df[col] = df[col].fillna('Unknown')
                logger.info(f"  Filled {missing_count} missing values in '{col}' with 'Unknown'")
        elif 'datetime' in str(df[col].dtype):
            # For datetime columns, forward fill
            df[col] = df[col].ffill()
            logger.info(f"  Forward filled {missing_count} missing datetime values in '{col}'")

    return df


def _handle_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> int:
    """
    Detect and cap outliers using IQR method.
    Returns the number of outliers handled.
    """
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, 0)
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    if outliers > 0:
        df[column] = df[column].clip(lower_bound, upper_bound)
    return outliers


def _convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    # Convert date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'day' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"  Converted '{col}' to datetime")
            except Exception as e:
                logger.warning(f"  Could not convert '{col}' to datetime: {e}")

    # Convert numeric columns stored as strings
    numeric_like = ['Sales', 'Profit', 'Discount', 'Quantity', 'Price', 'Cost']
    for col in numeric_like:
        if col in df.columns and df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col].str.replace('[$€£,]', '', regex=True), errors='coerce')
                logger.info(f"  Converted '{col}' to numeric")
            except Exception as e:
                logger.warning(f"  Could not convert '{col}' to numeric: {e}")

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline.
    Creates time-based, customer-based, and product-based features.
    """
    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING")
    logger.info("=" * 60)
    df = df.copy()

    # Time-based features
    logger.info("\n[1/4] Creating time-based features...")
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        df['Year'] = df['Order_Date'].dt.year
        df['Month'] = df['Order_Date'].dt.month
        df['Month_Name'] = df['Order_Date'].dt.month_name()
        df['Quarter'] = df['Order_Date'].dt.quarter
        df['Day'] = df['Order_Date'].dt.day
        df['Day_of_Week'] = df['Order_Date'].dt.dayofweek
        df['Day_Name'] = df['Order_Date'].dt.day_name()
        df['Week_of_Year'] = df['Order_Date'].dt.isocalendar().week.astype(int)
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        df['Is_Month_Start'] = df['Day'].apply(lambda x: 1 if x <= 7 else 0)
        df['Is_Month_End'] = df['Day'].apply(lambda x: 1 if x >= 21 else 0)
        df['Year_Month'] = df['Order_Date'].dt.to_period('M').astype(str)
        logger.info("  Created: Year, Month, Quarter, Day_of_Week, Week_of_Year, Is_Weekend, Year_Month")

        # Season
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        logger.info("  Created: Season")

        # Holiday season indicator
        df['Is_Holiday_Season'] = df['Month'].isin([11, 12]).astype(int)
        logger.info("  Created: Is_Holiday_Season")

    # Calculated metrics
    logger.info("\n[2/4] Creating calculated metrics...")
    if 'Sales' in df.columns and 'Quantity' in df.columns:
        df['Unit_Price'] = (df['Sales'] / df['Quantity']).round(2)
        logger.info("  Created: Unit_Price")

    if 'Sales' in df.columns and 'Profit' in df.columns:
        df['Profit_Margin_Pct'] = ((df['Profit'] / df['Sales']) * 100).round(2)
        logger.info("  Created: Profit_Margin_Pct")
        # Flag unprofitable orders
        df['Is_Profitable'] = (df['Profit'] > 0).astype(int)
        logger.info("  Created: Is_Profitable")

    if 'Sales' in df.columns and 'Discount' in df.columns:
        df['Discounted_Price'] = (df['Sales'] * (1 - df['Discount'])).round(2)
        logger.info("  Created: Discounted_Price")

    if 'Sales' in df.columns:
        # Sales per day bucket
        df['Sales_Bucket'] = pd.cut(
            df['Sales'],
            bins=[0, 50, 200, 500, 1000, 5000, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Premium']
        )
        logger.info("  Created: Sales_Bucket")

    # Customer-based features
    logger.info("\n[3/4] Creating customer-based features...")
    if 'Customer_ID' in df.columns:
        # These will be computed per group
        customer_stats = df.groupby('Customer_ID').agg(
            Customer_Total_Spent=('Sales', 'sum'),
            Customer_Total_Orders=('Order_ID', 'nunique'),
            Customer_Avg_Order_Value=('Sales', 'mean'),
            Customer_Total_Profit=('Profit', 'sum'),
            Customer_Avg_Discount=('Discount', 'mean'),
            Customer_First_Purchase=('Order_Date', 'min'),
            Customer_Last_Purchase=('Order_Date', 'max')
        ).reset_index()

        # Customer tenure and recency
        customer_stats['Customer_Tenure_Days'] = (
            customer_stats['Customer_Last_Purchase'] - customer_stats['Customer_First_Purchase']
        ).dt.days

        customer_stats['Customer_Recency_Days'] = (
            pd.Timestamp.now() - customer_stats['Customer_Last_Purchase']
        ).dt.days

        # Customer segment based on total spend
        customer_stats['Customer_Segment'] = pd.qcut(
            customer_stats['Customer_Total_Spent'],
            q=4,
            labels=['Bronze', 'Silver', 'Gold', 'Platinum']
        )

        df = df.merge(customer_stats, on='Customer_ID', how='left')
        logger.info("  Created: Customer segments, tenure, recency, and spending metrics")

    # Product-based features
    logger.info("\n[4/4] Creating product-based features...")
    if 'Product_Name' in df.columns:
        product_stats = df.groupby('Product_Name').agg(
            Product_Total_Sales=('Sales', 'sum'),
            Product_Total_Quantity=('Quantity', 'sum'),
            Product_Avg_Price=('Sales', 'mean'),
            Product_Total_Profit=('Profit', 'sum'),
            Product_Order_Count=('Order_ID', 'nunique')
        ).reset_index()

        # Product performance category
        product_stats['Product_Performance'] = pd.qcut(
            product_stats['Product_Total_Sales'],
            q=4,
            labels=['Low', 'Medium', 'High', 'Top']
        )

        df = df.merge(product_stats, on='Product_Name', how='left')
        logger.info("  Created: Product performance categories and metrics")

    logger.info(f"\nFeature engineering complete. Total columns: {len(df.columns)}")
    return df


def save_processed_data(df: pd.DataFrame, filename: str = 'sales_dataset_clean.csv'):
    """Save the processed dataset to disk."""
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"Processed dataset saved to {path} (compressed)")
    return path


def run_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete preprocessing and feature engineering pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE PREPROCESSING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Clean the dataset
    df_clean = clean_dataset(df)

    # Step 2: Feature engineering
    df_features = create_features(df_clean)

    # Step 3: Save
    path = save_processed_data(df_features)

    logger.info(f"\nPreprocessing complete. {len(df_features):,} records with {len(df_features.columns)} features")
    return df_features


if __name__ == '__main__':
    # Test the pipeline
    from data_loader import load_dataset
    raw_df = load_dataset()
    processed_df = run_preprocessing_pipeline(raw_df)
    print(f"\nProcessed data shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
