"""
Analysis Module - Comprehensive sales data analysis.
Performs trend analysis, customer insights, and KPI generation.
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


class SalesAnalyzer:
    """
    Performs comprehensive sales analytics including trend analysis,
    customer insights, product performance, and KPI generation.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Ensure datetime
        if 'Order_Date' in self.df.columns:
            self.df['Order_Date'] = pd.to_datetime(self.df['Order_Date'], errors='coerce')

    def analyze_sales_trends(self) -> dict:
        """Analyze sales trends over time."""
        self.logger.info("Analyzing sales trends...")

        # Daily sales
        daily_sales = self.df.groupby('Order_Date').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique')
        ).reset_index().sort_values('Order_Date')

        # Monthly trends
        monthly_trends = self.df.groupby('Year_Month').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Avg_Order_Value=('Sales', 'mean'),
            Order_Count=('Order_ID', 'nunique'),
            Customer_Count=('Customer_ID', 'nunique')
        ).reset_index().sort_values('Year_Month')

        # Quarterly trends
        quarterly_trends = self.df.groupby(['Year', 'Quarter']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique')
        ).reset_index().sort_values(['Year', 'Quarter'])

        # Year-over-Year growth
        yearly = self.df.groupby('Year').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).reset_index()
        yearly['Sales_Growth'] = yearly['Total_Sales'].pct_change() * 100
        yearly['Profit_Growth'] = yearly['Total_Profit'].pct_change() * 100

        # Day of week analysis
        dow_analysis = self.df.groupby('Day_Name').agg(
            Total_Sales=('Sales', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Sales=('Sales', 'mean')
        ).reset_index()

        # Day order for DOW
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_analysis['Day_Order'] = dow_analysis['Day_Name'].map(
            {d: i for i, d in enumerate(dow_order)}
        )
        dow_analysis = dow_analysis.sort_values('Day_Order')

        # Seasonal analysis
        seasonal = self.df.groupby('Season').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique')
        ).reset_index()

        return {
            'daily_sales': daily_sales,
            'monthly_trends': monthly_trends,
            'quarterly_trends': quarterly_trends,
            'yearly_analysis': yearly,
            'day_of_week_analysis': dow_analysis,
            'seasonal_analysis': seasonal
        }

    def analyze_customers(self) -> dict:
        """Analyze customer behavior and segments."""
        self.logger.info("Analyzing customer behavior...")

        customer_analysis = self.df.groupby(['Customer_ID', 'Customer_Name', 'Customer_Segment']).agg(
            Total_Spent=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Order_Value=('Sales', 'mean'),
            Total_Quantity=('Quantity', 'sum'),
            Avg_Discount=('Discount', 'mean'),
            First_Purchase=('Order_Date', 'min'),
            Last_Purchase=('Order_Date', 'max'),
            Preferred_Category=('Category', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A'),
            Preferred_Region=('Region', lambda x: x.mode().iloc[0] if not x.mode().empty else 'N/A')
        ).reset_index()

        # Recency, Frequency, Monetary (RFM) analysis
        max_date = self.df['Order_Date'].max()
        customer_analysis['Recency_Days'] = (max_date - customer_analysis['Last_Purchase']).dt.days
        customer_analysis['Frequency'] = customer_analysis['Order_Count']
        customer_analysis['Monetary'] = customer_analysis['Total_Spent']

        # RFM Scores (1-5)
        customer_analysis['R_Score'] = pd.qcut(customer_analysis['Recency_Days'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        customer_analysis['F_Score'] = pd.qcut(customer_analysis['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        customer_analysis['M_Score'] = pd.qcut(customer_analysis['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # Convert scores to numeric
        for col in ['R_Score', 'F_Score', 'M_Score']:
            customer_analysis[col] = pd.to_numeric(customer_analysis[col], errors='coerce')

        customer_analysis['RFM_Score'] = (
            customer_analysis['R_Score'].fillna(0).astype(int) +
            customer_analysis['F_Score'].fillna(0).astype(int) +
            customer_analysis['M_Score'].fillna(0).astype(int)
        )

        # Customer lifetime value
        customer_analysis['CLV'] = (
            customer_analysis['Avg_Order_Value'] * customer_analysis['Order_Count']
        )

        # Top customers by spend
        top_customers = customer_analysis.nlargest(10, 'Total_Spent')

        # Customer segment distribution
        segment_dist = self.df['Customer_Segment'].value_counts().reset_index()
        segment_dist.columns = ['Segment', 'Count']

        return {
            'customer_profiles': customer_analysis,
            'top_customers': top_customers,
            'segment_distribution': segment_dist,
            'rfm_analysis': customer_analysis[['Customer_ID', 'Customer_Name', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']]
        }

    def analyze_regions(self) -> dict:
        """Analyze regional performance."""
        self.logger.info("Analyzing regional performance...")

        # Region performance
        region_perf = self.df.groupby(['Region', 'Country', 'State']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Customer_Count=('Customer_ID', 'nunique'),
            Avg_Order_Value=('Sales', 'mean'),
            Avg_Profit_Margin=('Profit_Margin_Pct', 'mean'),
            Total_Quantity=('Quantity', 'sum')
        ).reset_index().sort_values('Total_Sales', ascending=False)

        # Region over time
        region_monthly = self.df.groupby(['Region', 'Year_Month']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).reset_index().sort_values(['Region', 'Year_Month'])

        # Profitability by region
        region_profitability = self.df.groupby('Region').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Avg_Margin=('Profit_Margin_Pct', 'mean'),
            Total_Orders=('Order_ID', 'nunique')
        ).reset_index()
        region_profitability['Profit_Ratio'] = (
            region_profitability['Total_Profit'] / region_profitability['Total_Sales'] * 100
        )

        # Top states
        top_states = self.df.groupby('State').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).reset_index().nlargest(10, 'Total_Sales')

        return {
            'region_performance': region_perf,
            'region_monthly': region_monthly,
            'region_profitability': region_profitability,
            'top_states': top_states
        }

    def analyze_products(self) -> dict:
        """Analyze product performance."""
        self.logger.info("Analyzing product performance...")

        # Product performance
        product_perf = self.df.groupby(['Product_Name', 'Category', 'Sub_Category', 'Product_Performance']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Price=('Sales', 'mean'),
            Avg_Profit_Margin=('Profit_Margin_Pct', 'mean'),
            Total_Quantity=('Quantity', 'sum'),
            Avg_Discount=('Discount', 'mean')
        ).reset_index().sort_values('Total_Sales', ascending=False)

        # Top products
        top_products = product_perf.nlargest(15, 'Total_Sales')

        # Category summary
        category_summary = self.df.groupby(['Category', 'Sub_Category']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Product_Count=('Product_Name', 'nunique'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Margin=('Profit_Margin_Pct', 'mean')
        ).reset_index().sort_values('Total_Sales', ascending=False)

        # Performance distribution
        perf_dist = self.df['Product_Performance'].value_counts().reset_index()
        perf_dist.columns = ['Performance', 'Count']

        return {
            'product_performance': product_perf,
            'top_products': top_products,
            'category_summary': category_summary,
            'performance_distribution': perf_dist
        }

    def analyze_profit(self) -> dict:
        """Analyze profit trends and drivers."""
        self.logger.info("Analyzing profit drivers...")

        # Profit by category
        profit_by_category = self.df.groupby('Category').agg(
            Total_Profit=('Profit', 'sum'),
            Total_Sales=('Sales', 'sum'),
            Avg_Margin=('Profit_Margin_Pct', 'mean'),
            Order_Count=('Order_ID', 'nunique')
        ).reset_index().sort_values('Total_Profit', ascending=False)

        # Profit vs Discount analysis
        profit_discount = self.df.groupby(pd.cut(self.df['Discount'], bins=np.arange(0, 0.8, 0.1))).agg(
            Total_Profit=('Profit', 'sum'),
            Total_Sales=('Sales', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Margin=('Profit_Margin_Pct', 'mean')
        ).reset_index()
        profit_discount.columns = ['Discount_Range', 'Total_Profit', 'Total_Sales', 'Order_Count', 'Avg_Margin']
        profit_discount['Discount_Range'] = profit_discount['Discount_Range'].astype(str)

        # Monthly profit trends
        monthly_profit = self.df.groupby('Year_Month').agg(
            Total_Profit=('Profit', 'sum'),
            Total_Sales=('Sales', 'sum'),
            Avg_Margin=('Profit_Margin_Pct', 'mean')
        ).reset_index().sort_values('Year_Month')

        # Most profitable products
        profitable_products = self.df.groupby('Product_Name').agg(
            Total_Profit=('Profit', 'sum'),
            Total_Sales=('Sales', 'sum'),
            Avg_Margin=('Profit_Margin_Pct', 'mean')
        ).reset_index().nlargest(10, 'Total_Profit')

        # Loss-making products
        loss_products = self.df.groupby('Product_Name').agg(
            Total_Profit=('Profit', 'sum'),
            Total_Sales=('Sales', 'sum')
        ).reset_index()
        loss_products = loss_products[loss_products['Total_Profit'] < 0].sort_values('Total_Profit')

        return {
            'profit_by_category': profit_by_category,
            'profit_discount_analysis': profit_discount,
            'monthly_profit_trends': monthly_profit,
            'most_profitable_products': profitable_products,
            'loss_making_products': loss_products
        }

    def analyze_categories(self) -> dict:
        """Perform detailed category analysis."""
        self.logger.info("Analyzing categories...")

        # Overall category stats
        category_stats = self.df.groupby('Category').agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Product_Count=('Product_Name', 'nunique'),
            Customer_Count=('Customer_ID', 'nunique'),
            Avg_Order_Value=('Sales', 'mean'),
            Avg_Margin=('Profit_Margin_Pct', 'mean'),
            Avg_Discount=('Discount', 'mean'),
            Total_Quantity=('Quantity', 'sum')
        ).reset_index().sort_values('Total_Sales', ascending=False)

        # Category market share
        total_sales = category_stats['Total_Sales'].sum()
        category_stats['Market_Share_Pct'] = (category_stats['Total_Sales'] / total_sales * 100).round(2)

        # Sub-category breakdown
        subcategory_stats = self.df.groupby(['Category', 'Sub_Category']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum'),
            Order_Count=('Order_ID', 'nunique'),
            Avg_Margin=('Profit_Margin_Pct', 'mean')
        ).reset_index().sort_values(['Category', 'Total_Sales'], ascending=[True, False])

        # Category trends over time
        category_trends = self.df.groupby(['Year_Month', 'Category']).agg(
            Total_Sales=('Sales', 'sum')
        ).reset_index().sort_values(['Year_Month', 'Category'])

        # Category by region
        category_region = self.df.groupby(['Category', 'Region']).agg(
            Total_Sales=('Sales', 'sum'),
            Total_Profit=('Profit', 'sum')
        ).reset_index().sort_values(['Category', 'Total_Sales'], ascending=[True, False])

        return {
            'category_stats': category_stats,
            'subcategory_stats': subcategory_stats,
            'category_trends': category_trends,
            'category_region': category_region
        }

    def analyze_revenue(self) -> dict:
        """Perform detailed revenue analysis."""
        self.logger.info("Analyzing revenue patterns...")

        # Monthly revenue
        monthly_revenue = self.df.groupby('Year_Month').agg(
            Revenue=('Sales', 'sum'),
            Profit=('Profit', 'sum'),
            Orders=('Order_ID', 'nunique')
        ).reset_index().sort_values('Year_Month')

        # Revenue by payment mode
        revenue_payment = self.df.groupby('Payment_Mode').agg(
            Revenue=('Sales', 'sum'),
            Orders=('Order_ID', 'nunique')
        ).reset_index().sort_values('Revenue', ascending=False)

        # Revenue by customer segment
        revenue_segment = self.df.groupby('Customer_Segment').agg(
            Revenue=('Sales', 'sum'),
            Customers=('Customer_ID', 'nunique'),
            Avg_Revenue=('Sales', 'mean')
        ).reset_index().sort_values('Revenue', ascending=False)

        # Revenue distribution
        revenue_bins = self.df['Sales_Bucket'].value_counts().reset_index()
        revenue_bins.columns = ['Bucket', 'Count']

        # Cumulative revenue
        daily_revenue = self.df.groupby('Order_Date').agg(
            Daily_Revenue=('Sales', 'sum')
        ).reset_index().sort_values('Order_Date')
        daily_revenue['Cumulative_Revenue'] = daily_revenue['Daily_Revenue'].cumsum()
        daily_revenue['Moving_Avg_7d'] = daily_revenue['Daily_Revenue'].rolling(window=7).mean()

        return {
            'monthly_revenue': monthly_revenue,
            'revenue_by_payment': revenue_payment,
            'revenue_by_segment': revenue_segment,
            'revenue_distribution': revenue_bins,
            'daily_revenue_trend': daily_revenue
        }

    def generate_kpis(self) -> dict:
        """Generate comprehensive KPIs."""
        self.logger.info("Generating KPIs...")

        df = self.df
        total_sales = df['Sales'].sum()
        total_profit = df['Profit'].sum()
        total_orders = df['Order_ID'].nunique()
        total_customers = df['Customer_ID'].nunique()
        total_products = df['Product_Name'].nunique()

        kpis = {
            # Revenue KPIs
            'Total_Revenue': round(total_sales, 2),
            'Average_Order_Value': round(df['Sales'].mean(), 2),
            'Revenue_per_Customer': round(total_sales / total_customers, 2) if total_customers > 0 else 0,
            'Revenue_per_Product': round(total_sales / total_products, 2) if total_products > 0 else 0,

            # Profit KPIs
            'Total_Profit': round(total_profit, 2),
            'Profit_Margin': round((total_profit / total_sales * 100), 2) if total_sales > 0 else 0,
            'Profit_per_Order': round(total_profit / total_orders, 2) if total_orders > 0 else 0,
            'Profit_per_Customer': round(total_profit / total_customers, 2) if total_customers > 0 else 0,

            # Order KPIs
            'Total_Orders': total_orders,
            'Orders_per_Customer': round(total_orders / total_customers, 2) if total_customers > 0 else 0,
            'Items_per_Order': round(df['Quantity'].sum() / total_orders, 1) if total_orders > 0 else 0,

            # Customer KPIs
            'Total_Customers': total_customers,
            'New_Customers': total_customers,

            # Product KPIs
            'Total_Products': total_products,
            'Products_per_Order': round(df.groupby('Order_ID')['Product_Name'].nunique().mean(), 1),

            # Discount KPIs
            'Average_Discount': round(df['Discount'].mean(), 3),
            'Total_Discount_Impact': round((df['Sales'] * df['Discount']).sum(), 2),

            # Performance KPIs
            'Total_Quantity_Sold': int(df['Quantity'].sum()),
            'Avg_Daily_Sales': round(df.groupby('Order_Date')['Sales'].sum().mean(), 2),
            'Peak_Sales_Day': round(df.groupby('Order_Date')['Sales'].sum().max(), 2),

            # Date range
            'Date_Start': df['Order_Date'].min(),
            'Date_End': df['Order_Date'].max(),
        }

        return kpis

    def run_all_analyses(self) -> dict:
        """Run all analysis modules."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING COMPREHENSIVE ANALYSIS")
        self.logger.info("=" * 60)

        results = {
            'sales_trends': self.analyze_sales_trends(),
            'customer_analysis': self.analyze_customers(),
            'regional_analysis': self.analyze_regions(),
            'product_analysis': self.analyze_products(),
            'profit_analysis': self.analyze_profit(),
            'category_analysis': self.analyze_categories(),
            'revenue_analysis': self.analyze_revenue(),
            'kpis': self.generate_kpis()
        }

        self.logger.info("\nAnalysis complete!")
        return results


if __name__ == '__main__':
    from data_loader import load_dataset
    from preprocessing import run_preprocessing_pipeline

    df = load_dataset()
    df = run_preprocessing_pipeline(df)
    analyzer = SalesAnalyzer(df)
    results = analyzer.run_all_analyses()

    print("\n=== KEY KPIs ===")
    for key, value in results['kpis'].items():
        print(f"  {key}: {value}")
