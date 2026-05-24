"""
AI-Powered Sales Analytics Dashboard
Main entry point - Orchestrates the complete analytics pipeline.

Usage:
    python main.py           # Run full pipeline
    python main.py --dashboard  # Launch the Streamlit dashboard
    python main.py --report     # Generate reports only
    python main.py --help       # Show help
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent


def print_banner():
    """Print the project banner using ASCII-safe characters."""
    banner = """
{sep}
    ** AI-Powered Sales Analytics Dashboard **
{sep}
    Data Analyst / Data Science Portfolio Project
    Built with Python, Machine Learning & Streamlit

    Features:
    * Automated data loading & preprocessing
    * Comprehensive sales analytics
    * Machine Learning forecasting (Prophet/XGBoost)
    * Interactive Streamlit dashboard
    * SQL analytics engine
    * PDF/CSV report generation
    * AI-powered business insights
    * Anomaly detection
{sep}
""".format(sep='=' * 60)
    print(banner)
    sys.stdout.flush()


def run_pipeline():
    """Run the complete data pipeline."""
    logger.info("=" * 60)
    logger.info("STARTING COMPLETE ANALYTICS PIPELINE")
    logger.info("=" * 60)
    start_time = datetime.now()

    # Step 1: Data Loading
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 60)
    from data_loader import load_dataset, get_dataset_info
    df = load_dataset()
    info = get_dataset_info(df)
    logger.info(f"  Loaded {info['rows']:,} records with {info['columns']} columns")

    # Step 2: Data Preprocessing
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    logger.info("=" * 60)
    from preprocessing import run_preprocessing_pipeline
    df = run_preprocessing_pipeline(df)
    logger.info(f"  Processed data: {len(df):,} records with {len(df.columns):,} features")

    # Step 3: SQL Analytics
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: SQL ANALYTICS ENGINE")
    logger.info("=" * 60)
    from sql_engine import create_analytics_database
    engine = create_analytics_database(df)

    # Test queries
    logger.info("\n  Running SQL analytics queries...")
    queries = {
        "Top Customers": lambda: engine.top_customers(5),
        "Best Products": lambda: engine.best_products(5),
        "Category Performance": lambda: engine.category_performance(),
        "Regional Analysis": lambda: engine.regional_analysis(),
        "Customer Segments": lambda: engine.customer_segment_analysis(),
        "Year-over-Year": lambda: engine.year_over_year_growth(),
        "Payment Mode": lambda: engine.payment_mode_analysis(),
        "KPIs": lambda: engine.get_kpi_summary(),
    }

    for name, query_fn in queries.items():
        try:
            result = query_fn()
            if isinstance(result, dict):
                logger.info(f"  V {name}")
            else:
                logger.info(f"  V {name}: {len(result)} rows")
        except Exception as e:
            logger.warning(f"  X {name}: {e}")

    # Step 4: Comprehensive Analysis
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: COMPREHENSIVE DATA ANALYSIS")
    logger.info("=" * 60)
    from analysis import SalesAnalyzer
    analyzer = SalesAnalyzer(df)
    analysis_results = analyzer.run_all_analyses()

    # Print key KPIs
    kpis = analysis_results['kpis']
    logger.info("\n  Key Performance Indicators:")
    logger.info(f"    Total Revenue:    ${kpis['Total_Revenue']:>12,.2f}")
    logger.info(f"    Total Profit:     ${kpis['Total_Profit']:>12,.2f}")
    logger.info(f"    Profit Margin:    {kpis['Profit_Margin']:>10.1f}%")
    logger.info(f"    Total Orders:     {kpis['Total_Orders']:>12,}")
    logger.info(f"    Total Customers:  {kpis['Total_Customers']:>12,}")
    logger.info(f"    Avg Order Value:  ${kpis['Average_Order_Value']:>12,.2f}")

    # Step 5: Machine Learning Forecasting
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: ML FORECASTING")
    logger.info("=" * 60)
    from forecasting import SalesForecaster
    forecaster = SalesForecaster(df)
    forecasting_results = forecaster.run_all_forecasts()

    # Print forecasting results
    for model_name in ['prophet', 'xgboost']:
        result = forecasting_results.get(model_name, {})
        accuracy = result.get('accuracy', {})
        if accuracy:
            logger.info(f"\n  {model_name.upper()} Model Performance:")
            for metric, value in accuracy.items():
                if isinstance(value, (int, float)):
                    logger.info(f"    {metric}: {value:.2f}")

    # Print insights
    insights = forecasting_results.get('insights', [])
    logger.info(f"\n  Generated {len(insights)} business insights")

    # Anomalies
    anomalies = forecasting_results.get('anomalies', None)
    if anomalies is not None:
        logger.info(f"  Detected {len(anomalies)} anomalies in the data")

    # Step 6: Report Generation
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: REPORT GENERATION")
    logger.info("=" * 60)
    from report_generator import ReportGenerator
    report_gen = ReportGenerator(df, analysis_results)
    reports = report_gen.generate_all_reports(anomalies)

    for report_name, path in reports.items():
        if path:
            logger.info(f"  V {report_name}: {path}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info(f"** PIPELINE COMPLETE! ({elapsed:.1f} seconds)")
    logger.info("=" * 60)
    logger.info(f"\n  Dashboard ready: streamlit run dashboard/app.py")
    logger.info(f"  Reports saved to: reports/")
    logger.info(f"  Models saved to: models/")
    logger.info(f"  Database: database.db")

    return df, analysis_results, forecasting_results, reports


def launch_dashboard():
    """Launch the Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard...")
    import subprocess
    import sys

    dashboard_path = PROJECT_ROOT / 'dashboard' / 'app.py'
    cmd = [sys.executable, '-m', 'streamlit', 'run', str(dashboard_path), '--server.port=8501']

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("Dashboard will be available at http://localhost:8501")

    subprocess.run(cmd)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AI-Powered Sales Analytics Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run the complete pipeline
  python main.py --dashboard   Launch the interactive dashboard
  python main.py --report      Generate reports only
  python main.py --data-only   Load and process data only
        """
    )

    parser.add_argument(
        '--dashboard', action='store_true',
        help='Launch the Streamlit dashboard'
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Generate reports only (skip dashboard)'
    )
    parser.add_argument(
        '--data-only', action='store_true',
        help='Only load and process data (skip analysis)'
    )

    args = parser.parse_args()

    print_banner()

    if args.dashboard:
        # Quick init then launch dashboard
        from data_loader import load_dataset
        from preprocessing import run_preprocessing_pipeline
        logger.info("Quick-loading data for dashboard...")
        df = load_dataset()
        df = run_preprocessing_pipeline(df)

        from sql_engine import create_analytics_database
        create_analytics_database(df)

        launch_dashboard()
    elif args.report:
        run_pipeline()
    elif args.data_only:
        from data_loader import load_dataset
        from preprocessing import run_preprocessing_pipeline
        df = load_dataset()
        df = run_preprocessing_pipeline(df)
        logger.info(f"Data ready: {len(df):,} records with {len(df.columns):,} features")
    else:
        run_pipeline()


if __name__ == '__main__':
    main()
