"""
SQL Analytics Engine - Creates and manages SQLite database for sales analytics.
Provides business intelligence queries for top customers, products, and trends.
"""

import logging
import sqlite3
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
DB_PATH = PROJECT_ROOT / 'database.db'


class SQLAnalyticsEngine:
    """
    SQL Analytics Engine for business intelligence queries.
    Manages SQLite database and provides analytical query methods.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing SQL engine with database: {self.db_path}")

    def create_connection(self) -> sqlite3.Connection:
        """Create a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def create_tables(self, conn: sqlite3.Connection):
        """Create optimized database tables with indexes."""
        cursor = conn.cursor()

        # Main orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                Order_ID TEXT PRIMARY KEY,
                Customer_Name TEXT,
                Customer_ID TEXT,
                Product_Name TEXT,
                Category TEXT,
                Sub_Category TEXT,
                Sales REAL,
                Profit REAL,
                Discount REAL,
                Quantity INTEGER,
                Order_Date DATE,
                Region TEXT,
                Country TEXT,
                State TEXT,
                Payment_Mode TEXT,
                Year INTEGER,
                Month INTEGER,
                Quarter INTEGER,
                Day_of_Week INTEGER,
                Month_Name TEXT,
                Season TEXT,
                Is_Holiday_Season INTEGER,
                Year_Month TEXT,
                Unit_Price REAL,
                Profit_Margin_Pct REAL,
                Is_Profitable INTEGER,
                Customer_Total_Spent REAL,
                Customer_Total_Orders INTEGER,
                Customer_Segment TEXT,
                Product_Performance TEXT
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_date ON orders(Order_Date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_customer ON orders(Customer_ID)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON orders(Category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_region ON orders(Region)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_year_month ON orders(Year_Month)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_product ON orders(Product_Name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_country ON orders(Country)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_segment ON orders(Customer_Segment)')

        conn.commit()
        self.logger.info("Database tables and indexes created successfully")

    def load_data(self, df: pd.DataFrame):
        """Load DataFrame into the database."""
        self.logger.info(f"Loading {len(df):,} records into database...")
        conn = self.create_connection()
        self.create_tables(conn)

        # Filter to only columns that exist in the table
        cols = [c for c in [
            'Order_ID', 'Customer_Name', 'Customer_ID', 'Product_Name',
            'Category', 'Sub_Category', 'Sales', 'Profit', 'Discount',
            'Quantity', 'Order_Date', 'Region', 'Country', 'State', 'Payment_Mode',
            'Year', 'Month', 'Quarter', 'Day_of_Week', 'Month_Name',
            'Season', 'Is_Holiday_Season', 'Year_Month', 'Unit_Price',
            'Profit_Margin_Pct', 'Is_Profitable', 'Customer_Total_Spent',
            'Customer_Total_Orders', 'Customer_Segment', 'Product_Performance'
        ] if c in df.columns]

        df_subset = df[cols].copy()

        # Drop existing data and re-insert (handles re-runs)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM orders")
        conn.commit()

        # Insert data in chunks for compatibility (avoids SQL variable limit)
        chunksize = 100
        total = len(df_subset)
        for i in range(0, total, chunksize):
            chunk = df_subset.iloc[i:i + chunksize]
            chunk.to_sql('orders', conn, if_exists='append', index=False, method=None)
            if (i + chunksize) % 500 == 0 or (i + chunksize) >= total:
                conn.commit()
            self.logger.debug(f"  Loaded rows {i:,} to {min(i + chunksize, total):,}")

        conn.commit()
        conn.close()
        self.logger.info(f"Successfully loaded {len(df):,} records into database")

    def execute_query(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        try:
            conn = self.create_connection()
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            self.logger.error(f"Query: {query}")
            raise

    # ==========================================================================
    # BUSINESS INTELLIGENCE QUERIES
    # ==========================================================================

    def top_customers(self, limit: int = 10) -> pd.DataFrame:
        """Get top customers by total revenue."""
        query = '''
            SELECT
                Customer_ID,
                Customer_Name,
                Customer_Segment,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                ROUND(SUM(Quantity), 0) AS Total_Items
            FROM orders
            GROUP BY Customer_ID
            ORDER BY Total_Revenue DESC
            LIMIT ?
        '''
        return self.execute_query(query, (limit,))

    def best_products(self, limit: int = 10) -> pd.DataFrame:
        """Get best-selling products by revenue and profit."""
        query = '''
            SELECT
                Product_Name,
                Category,
                Sub_Category,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                ROUND(AVG(Discount), 3) AS Avg_Discount,
                SUM(Quantity) AS Total_Units_Sold
            FROM orders
            GROUP BY Product_Name
            ORDER BY Total_Revenue DESC
            LIMIT ?
        '''
        return self.execute_query(query, (limit,))

    def profit_trends(self) -> pd.DataFrame:
        """Get monthly profit trends."""
        query = '''
            SELECT
                Year_Month,
                Year,
                Month,
                Month_Name,
                ROUND(SUM(Sales), 2) AS Total_Sales,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(AVG(Discount), 3) AS Avg_Discount
            FROM orders
            GROUP BY Year_Month
            ORDER BY Year, Month
        '''
        return self.execute_query(query)

    def regional_analysis(self) -> pd.DataFrame:
        """Get regional sales performance."""
        query = '''
            SELECT
                Region,
                Country,
                ROUND(SUM(Sales), 2) AS Total_Sales,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                COUNT(DISTINCT Customer_ID) AS Unique_Customers,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(SUM(Sales) * 100.0 / SUM(SUM(Sales)) OVER(), 2) AS Sales_Share_Pct
            FROM orders
            GROUP BY Region, Country
            ORDER BY Total_Sales DESC
        '''
        return self.execute_query(query)

    def monthly_analysis(self) -> pd.DataFrame:
        """Get comprehensive monthly analysis."""
        query = '''
            SELECT
                Year,
                Month,
                Month_Name,
                Quarter,
                ROUND(SUM(Sales), 2) AS Monthly_Revenue,
                ROUND(SUM(Profit), 2) AS Monthly_Profit,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                COUNT(DISTINCT Customer_ID) AS Unique_Customers,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(AVG(Discount), 3) AS Avg_Discount,
                SUM(Quantity) AS Total_Units_Sold
            FROM orders
            GROUP BY Year, Month
            ORDER BY Year, Month
        '''
        return self.execute_query(query)

    def category_performance(self) -> pd.DataFrame:
        """Get category and sub-category performance."""
        query = '''
            SELECT
                Category,
                Sub_Category,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                ROUND(AVG(Discount), 3) AS Avg_Discount,
                SUM(Quantity) AS Total_Units_Sold,
                ROUND(SUM(Sales) * 100.0 / SUM(SUM(Sales)) OVER(PARTITION BY Category), 2) AS Category_Share_Pct
            FROM orders
            GROUP BY Category, Sub_Category
            ORDER BY Total_Revenue DESC
        '''
        return self.execute_query(query)

    def customer_segment_analysis(self) -> pd.DataFrame:
        """Analyze customer segments."""
        query = '''
            SELECT
                Customer_Segment,
                COUNT(DISTINCT Customer_ID) AS Customer_Count,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                ROUND(AVG(Customer_Total_Orders), 1) AS Avg_Orders_Per_Customer,
                ROUND(SUM(Sales) * 100.0 / SUM(SUM(Sales)) OVER(), 2) AS Revenue_Share_Pct
            FROM orders
            GROUP BY Customer_Segment
            ORDER BY Total_Revenue DESC
        '''
        return self.execute_query(query)

    def payment_mode_analysis(self) -> pd.DataFrame:
        """Analyze sales by payment mode."""
        query = '''
            SELECT
                Payment_Mode,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                ROUND(SUM(Sales) * 100.0 / SUM(SUM(Sales)) OVER(), 2) AS Revenue_Share_Pct
            FROM orders
            GROUP BY Payment_Mode
            ORDER BY Total_Revenue DESC
        '''
        return self.execute_query(query)

    def seasonal_analysis(self) -> pd.DataFrame:
        """Analyze seasonal patterns."""
        query = '''
            SELECT
                Season,
                Year,
                ROUND(SUM(Sales), 2) AS Total_Sales,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                COUNT(DISTINCT Order_ID) AS Order_Count,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value
            FROM orders
            GROUP BY Season, Year
            ORDER BY Year, Season
        '''
        return self.execute_query(query)

    def year_over_year_growth(self) -> pd.DataFrame:
        """Calculate YoY growth metrics."""
        query = '''
            WITH yearly AS (
                SELECT
                    Year,
                    ROUND(SUM(Sales), 2) AS Total_Revenue,
                    ROUND(SUM(Profit), 2) AS Total_Profit,
                    COUNT(DISTINCT Order_ID) AS Order_Count,
                    COUNT(DISTINCT Customer_ID) AS Customer_Count
                FROM orders
                GROUP BY Year
            )
            SELECT
                Year,
                Total_Revenue,
                ROUND((Total_Revenue - LAG(Total_Revenue) OVER (ORDER BY Year)) / 
                      NULLIF(LAG(Total_Revenue) OVER (ORDER BY Year), 0) * 100, 2) AS Revenue_Growth_Pct,
                Total_Profit,
                ROUND((Total_Profit - LAG(Total_Profit) OVER (ORDER BY Year)) / 
                      NULLIF(LAG(Total_Profit) OVER (ORDER BY Year), 0) * 100, 2) AS Profit_Growth_Pct,
                Order_Count,
                Customer_Count
            FROM yearly
            ORDER BY Year
        '''
        return self.execute_query(query)

    def get_kpi_summary(self) -> dict:
        """Get overall KPI summary from database."""
        query = '''
            SELECT
                COUNT(DISTINCT Order_ID) AS Total_Orders,
                COUNT(DISTINCT Customer_ID) AS Total_Customers,
                COUNT(DISTINCT Product_Name) AS Total_Products,
                ROUND(SUM(Sales), 2) AS Total_Revenue,
                ROUND(SUM(Profit), 2) AS Total_Profit,
                ROUND(AVG(Profit_Margin_Pct), 2) AS Avg_Profit_Margin,
                ROUND(AVG(Sales), 2) AS Avg_Order_Value,
                ROUND(AVG(Discount), 3) AS Avg_Discount,
                SUM(Quantity) AS Total_Units_Sold,
                ROUND(AVG(Quantity), 1) AS Avg_Items_Per_Order
            FROM orders
        '''
        result = self.execute_query(query)
        return result.iloc[0].to_dict()

    def anomaly_detection_query(self) -> pd.DataFrame:
        """Detect anomalous orders using statistical methods.
        Uses SQLite-compatible SQL (STDEV not available, computed in Python).
        """
        # Compute stats in Python since SQLite lacks STDEV
        conn = self.create_connection()
        df = pd.read_sql_query("SELECT Sales, Profit, Discount, Order_ID, Customer_Name, Product_Name, Category, Quantity, Order_Date, Region, Payment_Mode FROM orders", conn)
        conn.close()

        avg_sales = df['Sales'].mean()
        std_sales = df['Sales'].std()
        threshold = avg_sales + 3 * std_sales

        anomalies = df[
            (df['Sales'] > threshold) |
            (df['Sales'] < 0) |
            (df['Discount'] > 0.7) |
            (df['Profit'] < -1000)
        ].copy()

        conditions = [
            anomalies['Sales'] > threshold,
            anomalies['Sales'] < 0,
            anomalies['Discount'] > 0.7,
            anomalies['Profit'] < -1000
        ]
        choices = ['High Value Anomaly', 'Negative Sales', 'Extreme Discount', 'Large Loss']
        anomalies['Anomaly_Type'] = np.select(conditions, choices, default='Normal')

        return anomalies.sort_values('Sales', ascending=False)


def create_analytics_database(df: pd.DataFrame) -> SQLAnalyticsEngine:
    """Create the analytics database and load data."""
    logger.info("=" * 60)
    logger.info("CREATING ANALYTICS DATABASE")
    logger.info("=" * 60)

    engine = SQLAnalyticsEngine()
    engine.load_data(df)

    # Test queries
    logger.info("\nRunning test queries...")
    test_queries = [
        ("Top Customers", engine.top_customers(5)),
        ("Best Products", engine.best_products(5)),
        ("Profit Trends", engine.profit_trends().head()),
        ("Regional Analysis", engine.regional_analysis()),
    ]

    for name, result in test_queries:
        logger.info(f"\n{name}:")
        logger.info(f"\n{result.to_string()}")

    logger.info(f"\nDatabase created successfully at: {engine.db_path}")
    return engine


if __name__ == '__main__':
    from data_loader import load_dataset
    from preprocessing import run_preprocessing_pipeline

    df = load_dataset()
    df = run_preprocessing_pipeline(df)
    engine = create_analytics_database(df)
