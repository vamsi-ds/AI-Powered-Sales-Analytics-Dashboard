"""
Data Loader Module - Handles dataset download and synthetic data generation.
Supports automatic download from Kaggle, CSV files, and fallback synthetic generation.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset column definitions
REQUIRED_COLUMNS = [
    'Order_ID', 'Customer_Name', 'Customer_ID', 'Product_Name',
    'Category', 'Sub_Category', 'Sales', 'Profit', 'Discount',
    'Quantity', 'Order_Date', 'Region', 'Country', 'State', 'Payment_Mode'
]

# Synthetic data generation parameters
NUM_CUSTOMERS = 500
NUM_PRODUCTS = 200
NUM_ORDERS = 5000
START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Realistic product catalog
PRODUCT_CATALOG = {
    'Technology': {
        'Phones': ['Apple iPhone 14 Pro Max', 'Samsung Galaxy S24 Ultra', 'Google Pixel 8 Pro',
                   'OnePlus 12', 'Xiaomi 14 Pro', 'iPhone 15', 'Samsung Galaxy Z Fold',
                   'Motorola Edge 40', 'Pixel 7a', 'Nothing Phone 2'],
        'Laptops': ['MacBook Pro 16" M3', 'Dell XPS 15', 'ThinkPad X1 Carbon',
                    'HP Spectre x360', 'ASUS ROG Zephyrus', 'MacBook Air M2',
                    'Microsoft Surface Laptop 5', 'LG Gram 17', 'Acer Swift 5', 'Razer Blade 15'],
        'Accessories': ['AirPods Pro 2', 'Samsung Galaxy Buds', 'Apple Watch Ultra',
                        'Logitech MX Master 3S', 'Anker Power Bank', 'Apple AirTag',
                        'JBL Flip 6', 'Sony WH-1000XM5', 'Belkin USB-C Hub', 'Casetify Phone Case'],
        'Software': ['Microsoft Office 365', 'Adobe Creative Suite', 'Salesforce CRM',
                     'Slack Enterprise', 'Tableau Desktop', 'Jira Software',
                     'Figma Professional', 'Notion Team Plan', 'Zoom Business', 'GitHub Enterprise']
    },
    'Furniture': {
        'Chairs': ['Herman Miller Aeron', 'Steelcase Gesture', 'Branch Ergonomic Chair',
                   'Autonomous ErgoChair Pro', 'HON Ignition 2.0', 'IKEA Markus',
                   'Serta Big and Tall', 'La-Z-Boy Executive', 'Hbada Office Chair', 'Amazon Basics Chair'],
        'Tables': ['Uplift Standing Desk', 'Jarvis Bamboo Desk', 'IKEA Bekant Corner Desk',
                   'Vari Electric Standing Desk', 'Flexispot E7', 'Branch Standing Desk',
                   'IKEA Malm Desk', 'Sauder Edge Water Desk', 'Herman Miller Desk', 'Fully Desk'],
        'Bookcases': ['IKEA Billy Bookcase', 'Sauder Select Bookcase', 'Sauder Barrister Bookcase',
                      'IKEA Kallax Shelf', 'Atlantic Oskar Media', 'Prepac Kingston Bookcase',
                      'MainStays Bookcase', 'Furinno Turn-N-Tube', 'South Shore Smart', 'Sauder Edge Bookcase'],
        'Furnishings': ['IKEA LED Floor Lamp', 'Amazon Basics Desk Lamp', 'Philips Hue Lightstrip',
                        'Honeywell Air Purifier', 'BALI Blinds', 'IKEA Fintorp Rack',
                        'SimpleHuman Trash Can', 'Vornado Space Heater', 'IKEA Skadis Pegboard', 'Alexa Echo Dot']
    },
    'Office Supplies': {
        'Paper': ['Hammermill Printer Paper', 'HP Printer Paper 8.5x11', 'Staples Copy Paper',
                  'Boise X-9 Copy Paper', 'Georgia-Pacific Paper', 'Southworth Resume Paper',
                  'Canon Photo Paper', 'Epson Presentation Paper', 'Neenah Cardstock', 'Mohawk Paper'],
        'Binders': ['Avery Heavy-Duty Binder', 'Staples D-ring Binder', 'Samsill View Binder',
                    'Cardinal Economy Binder', 'Five Star Zipper Binder', 'Wilson Jones Binder',
                    'BetterOffice Binder', 'Oxford Binder', 'Kokuyo Campus Binder', 'Union & Scale Binder'],
        'Art': ['Copic Marker Set', 'Prismacolor Colored Pencils', 'Faber-Castell Pencils',
                'Sharpie Permanent Markers', 'Crayola Supertips Set', 'Tombow Dual Brush Pens',
                'Strathmore Sketch Pad', 'Moleskine Art Notebook', 'Winsor & Newton Watercolor', 'Derwent Graphic Pencils'],
        'Storage': ['Fellowes Bankers Box', 'Sterilite Plastic Bins', 'IRIS File Tote',
                    'Bankers Box SmoothMove', 'Honeywell Security Case', 'Really Useful Box',
                    'Container Store Bins', 'Rubbermaid Totes', 'Greenmade Storage Bin', 'Akro-Mils Storage'],
        'Appliances': ['Breville Espresso Machine', 'Keurig K-Elite', 'Cuisinart Coffee Pot',
                       'Mr. Coffee Maker', 'Hamilton Beach Microwave', 'Nespresso VertuoNext',
                       'Breville Toaster Oven', 'Instant Pot Duo', 'KitchenAid Water Filter', 'Frigidaire Mini Fridge']
    }
}

# Realistic customer data
FIRST_NAMES = [
    'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
    'David', 'Elizabeth', 'William', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica',
    'Thomas', 'Sarah', 'Christopher', 'Karen', 'Charles', 'Lisa', 'Daniel', 'Nancy',
    'Matthew', 'Betty', 'Anthony', 'Margaret', 'Mark', 'Sandra', 'Donald', 'Ashley',
    'Steven', 'Dorothy', 'Paul', 'Kimberly', 'Andrew', 'Emily', 'Joshua', 'Donna',
    'Kenneth', 'Michelle', 'Kevin', 'Carol', 'Brian', 'Amanda', 'George', 'Melissa',
    'Timothy', 'Deborah'
]

LAST_NAMES = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
    'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
    'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson',
    'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez', 'Lewis', 'Robinson', 'Walker',
    'Young', 'Allen', 'King', 'Wright', 'Scott', 'Torres', 'Nguyen', 'Hill', 'Flores',
    'Green', 'Adams', 'Nelson', 'Baker', 'Hall', 'Rivera', 'Campbell', 'Mitchell', 'Carter'
]

REGIONS = ['East', 'West', 'Central', 'South']
COUNTRIES = ['United States', 'Canada', 'United Kingdom', 'Germany', 'Australia']
US_STATES = [
    'California', 'Texas', 'New York', 'Florida', 'Illinois', 'Pennsylvania', 'Ohio',
    'Georgia', 'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington',
    'Arizona', 'Massachusetts', 'Tennessee', 'Indiana', 'Missouri', 'Maryland', 'Wisconsin',
    'Colorado', 'Minnesota', 'South Carolina', 'Alabama', 'Louisiana', 'Kentucky',
    'Oregon', 'Oklahoma', 'Connecticut', 'Utah'
]

CANADA_PROVINCES = ['Ontario', 'Quebec', 'British Columbia', 'Alberta', 'Manitoba']
UK_REGIONS = ['England', 'Scotland', 'Wales', 'Northern Ireland']
GERMANY_STATES = ['Bavaria', 'North Rhine-Westphalia', 'Baden-Wurttemberg', 'Hesse', 'Berlin']
AUSTRALIA_STATES = ['New South Wales', 'Victoria', 'Queensland', 'Western Australia', 'South Australia']

PAYMENT_MODES = ['Credit Card', 'Debit Card', 'PayPal', 'Bank Transfer', 'Apple Pay', 'Google Pay']
COUNTRY_STATE_MAP = {
    'United States': US_STATES,
    'Canada': CANADA_PROVINCES,
    'United Kingdom': UK_REGIONS,
    'Germany': GERMANY_STATES,
    'Australia': AUSTRALIA_STATES
}


def generate_synthetic_dataset(num_orders: int = NUM_ORDERS) -> pd.DataFrame:
    """
    Generate a realistic synthetic sales dataset with business-appropriate patterns.

    Features:
    - Seasonal sales patterns (holiday spikes)
    - Realistic pricing and discounting
    - Customer segmentation
    - Geographic distribution
    - Multiple product categories
    """
    logger.info(f"Generating synthetic dataset with {num_orders} orders...")
    random.seed(42)
    np.random.seed(42)

    # Generate customers
    customers = []
    for i in range(NUM_CUSTOMERS):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        customers.append({
            'Customer_ID': f'CUST-{i+1:05d}',
            'Customer_Name': f'{first} {last}'
        })

    # Assign customer regions more realistically
    customer_regions = {}
    customer_countries = {}
    customer_states = {}
    for cust in customers:
        country = random.choices(
            COUNTRIES,
            weights=[0.5, 0.15, 0.15, 0.1, 0.1],  # US more likely
            k=1
        )[0]
        states = COUNTRY_STATE_MAP[country]
        state = random.choice(states)

        if country == 'United States':
            if state in ['California', 'Washington', 'Oregon']:
                region = 'West'
            elif state in ['New York', 'New Jersey', 'Pennsylvania', 'Massachusetts']:
                region = 'East'
            elif state in ['Texas', 'Florida', 'Georgia', 'North Carolina', 'South Carolina', 'Alabama', 'Louisiana', 'Tennessee', 'Kentucky', 'Virginia', 'Maryland']:
                region = 'South'
            else:
                region = 'Central'
        elif country == 'Canada':
            region = 'Central'
        elif country == 'United Kingdom':
            region = 'East'
        elif country == 'Germany':
            region = 'Central'
        else:  # Australia
            region = 'West'

        customer_regions[cust['Customer_ID']] = region
        customer_countries[cust['Customer_ID']] = country
        customer_states[cust['Customer_ID']] = state

    # Build product catalog flat list
    products = []
    for category, subcategories in PRODUCT_CATALOG.items():
        for sub_category, items in subcategories.items():
            for product in items:
                base_price = np.random.uniform(15, 2500)
                if category == 'Technology':
                    base_price = base_price * 1.5  # Tech is more expensive
                elif category == 'Furniture':
                    base_price = base_price * 1.2
                products.append({
                    'Product_Name': product,
                    'Category': category,
                    'Sub_Category': sub_category,
                    'Base_Price': round(base_price, 2)
                })

    # Generate orders with realistic patterns
    orders = []
    date_range = (END_DATE - START_DATE).days

    # Seasonal multipliers
    def get_seasonal_multiplier(date):
        """Apply seasonal patterns to sales."""
        month = date.month
        day = date.day

        # Holiday season boost (Nov-Dec)
        if month == 12:
            return np.random.uniform(1.3, 1.8)
        elif month == 11:
            return np.random.uniform(1.2, 1.5)
        # Back to school (Aug-Sept)
        elif month == 8:
            return np.random.uniform(1.1, 1.3)
        elif month == 9:
            return np.random.uniform(1.0, 1.2)
        # Summer slowdown (June-July)
        elif month in [6, 7]:
            return np.random.uniform(0.7, 0.9)
        # New year (Jan-Feb)
        elif month == 1:
            return np.random.uniform(0.8, 1.0)
        else:
            return np.random.uniform(0.9, 1.1)

    for i in range(num_orders):
        # Random date with seasonal weighting
        order_date = START_DATE + timedelta(
            days=random.randint(0, date_range)
        )
        seasonal_factor = get_seasonal_multiplier(order_date)

        cust = random.choice(customers)
        product = random.choice(products)

        # Base quantity with seasonal adjustment
        base_qty = int(np.random.poisson(2) + 1)
        quantity = max(1, int(base_qty * seasonal_factor))

        # Pricing with some randomness
        discount = 0
        if random.random() < 0.35:  # 35% chance of discount
            discount = round(random.choices(
                [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
                weights=[0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05],
                k=1
            )[0], 2)

        unit_price = product['Base_Price'] * (1 - discount)
        sales = round(unit_price * quantity, 2)

        # Profit calculation with realistic margins
        cost_ratio = random.choices(
            [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
            weights=[0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05],
            k=1
        )[0]
        cost = unit_price * cost_ratio
        profit = round((unit_price - cost) * quantity, 2)

        cust_id = cust['Customer_ID']
        payment = random.choice(PAYMENT_MODES)

        order = {
            'Order_ID': f'ORD-{i+1:06d}',
            'Customer_Name': cust['Customer_Name'],
            'Customer_ID': cust_id,
            'Product_Name': product['Product_Name'],
            'Category': product['Category'],
            'Sub_Category': product['Sub_Category'],
            'Sales': sales,
            'Profit': profit,
            'Discount': discount,
            'Quantity': quantity,
            'Order_Date': order_date.strftime('%Y-%m-%d'),
            'Region': customer_regions[cust_id],
            'Country': customer_countries[cust_id],
            'State': customer_states[cust_id],
            'Payment_Mode': payment
        }
        orders.append(order)

    df = pd.DataFrame(orders)
    logger.info(f"Generated {len(df):,} synthetic sales records")
    return df


def try_download_kaggle_dataset() -> pd.DataFrame:
    """
    Attempt to download a sales dataset from Kaggle.
    Falls back to other sources if Kaggle is unavailable.
    """
    try:
        logger.info("Attempting to download Kaggle sales dataset...")

        # Try multiple known sales datasets on Kaggle
        datasets = [
            "rohitsahoo/sales-forecasting",
            "anandshaw2001/super-store-dataset",
            "vivek468/super-store-dataset-final",
        ]

        for dataset in datasets:
            try:
                import kagglehub
                logger.info(f"Trying dataset: {dataset}")
                path = kagglehub.dataset_download(dataset)
                csv_files = list(Path(path).glob('*.csv'))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    logger.info(f"Successfully downloaded {dataset}")
                    # Check if required columns exist (with some flexibility)
                    existing_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
                    if len(existing_cols) >= 10:  # At least 10 of 15 required columns
                        return df
            except Exception as e:
                logger.warning(f"Failed to download {dataset}: {e}")
                continue

        logger.warning("All Kaggle downloads failed")
        return None

    except ImportError:
        logger.warning("kagglehub not installed, skipping Kaggle download")
        return None
    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        return None


def try_download_csv_source() -> pd.DataFrame:
    """Attempt to download a sales dataset from public CSV sources."""
    import requests

    urls = [
        "https://raw.githubusercontent.com/safesoft77/Sample-Superstore-Dataset/master/Sample%20-%20Superstore%20-%20Data.csv",
        "https://raw.githubusercontent.com/mattdelarosa/global-superstore-dataset/master/Global%20Superstore%20-%20Data.csv",
    ]

    for url in urls:
        try:
            logger.info(f"Attempting to download from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(RAW_DATA_DIR / 'temp_download.csv', 'wb') as f:
                f.write(response.content)
            df = pd.read_csv(RAW_DATA_DIR / 'temp_download.csv')
            existing_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
            if len(existing_cols) >= 10:
                logger.info(f"Successfully downloaded from {url}")
                return df
        except Exception as e:
            logger.warning(f"Failed to download from {url}: {e}")
            continue

    return None


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the dataframe has the required columns."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns: {missing_cols}")
        return False
    return True


def map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map alternative column names to standard names if needed."""
    column_mappings = {
        'Order ID': 'Order_ID',
        'Customer Name': 'Customer_Name',
        'Customer ID': 'Customer_ID',
        'Product Name': 'Product_Name',
        'Sub-Category': 'Sub_Category',
        'Sub Category': 'Sub_Category',
        'Order Date': 'Order_Date',
        'Order_Date (DateOrders)': 'Order_Date',
        'Payment Mode': 'Payment_Mode',
        'Payment_Mode': 'Payment_Mode',
    }

    df = df.rename(columns=column_mappings)
    return df


def load_dataset() -> pd.DataFrame:
    """
    Main entry point - loads a sales dataset from any available source.
    Priority: 1) Local CSV, 2) Kaggle download, 3) CSV source, 4) Synthetic generation
    """
    logger.info("=" * 60)
    logger.info("LOADING SALES DATASET")
    logger.info("=" * 60)

    # Check for existing raw data
    local_files = list(RAW_DATA_DIR.glob('*.csv'))
    if local_files:
        logger.info(f"Local CSV found: {local_files[0]}")
        df = pd.read_csv(local_files[0])
        df = map_columns(df)
        if validate_schema(df):
            logger.info("Local dataset validated successfully")
            return df
        logger.warning("Local dataset missing required columns, will try other sources")

    # Try downloading from various sources
    logger.info("No valid local dataset found. Attempting download...")

    # Try Kaggle
    df = try_download_kaggle_dataset()
    if df is not None:
        df = map_columns(df)
        if validate_schema(df):
            logger.info("Kaggle dataset validated successfully")
            save_raw(df)
            return df

    # Try CSV sources
    df = try_download_csv_source()
    if df is not None:
        df = map_columns(df)
        if validate_schema(df):
            logger.info("CSV source dataset validated successfully")
            save_raw(df)
            return df

    # Fallback: Generate synthetic data
    logger.info("=" * 60)
    logger.info("GENERATING SYNTHETIC DATASET")
    logger.info("Automatically generating realistic enterprise sales data...")
    logger.info("=" * 60)

    df = generate_synthetic_dataset()
    save_raw(df)
    logger.info(f"Synthetic dataset saved with {len(df):,} records")
    return df


def save_raw(df: pd.DataFrame):
    """Save raw dataset to disk."""
    raw_path = RAW_DATA_DIR / 'sales_dataset_raw.csv'
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw dataset saved to {raw_path}")


def get_dataset_info(df: pd.DataFrame) -> dict:
    """Get basic information about the dataset."""
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'date_range': None,
        'total_sales': None,
        'total_profit': None,
        'num_customers': None,
        'num_products': None,
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }

    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
        info['date_range'] = {
            'start': df['Order_Date'].min(),
            'end': df['Order_Date'].max()
        }

    if 'Sales' in df.columns:
        info['total_sales'] = df['Sales'].sum()
    if 'Profit' in df.columns:
        info['total_profit'] = df['Profit'].sum()
    if 'Customer_ID' in df.columns:
        info['num_customers'] = df['Customer_ID'].nunique()
    if 'Product_Name' in df.columns:
        info['num_products'] = df['Product_Name'].nunique()

    return info


if __name__ == '__main__':
    df = load_dataset()
    info = get_dataset_info(df)
    print("\nDataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
