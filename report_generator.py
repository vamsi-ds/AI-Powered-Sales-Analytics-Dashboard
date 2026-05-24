"""
Report Generator Module - Generates PDF reports, CSV exports, and AI business insights.
Provides comprehensive reporting capabilities for the analytics dashboard.
"""

import logging
import io
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent
REPORTS_DIR = PROJECT_ROOT / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)


class ReportGenerator:
    """
    Generates professional business reports including PDF exports,
    CSV exports, and automated business insights.
    """

    def __init__(self, df: pd.DataFrame, analysis_results: dict = None):
        self.df = df.copy()
        self.analysis_results = analysis_results or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def generate_csv_summary(self) -> str:
        """Generate a comprehensive CSV summary report."""
        self.logger.info("Generating CSV summary report...")

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(['AI-Powered Sales Analytics Dashboard - Summary Report'])
        writer.writerow([f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([])

        # Dataset overview
        writer.writerow(['DATASET OVERVIEW'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Records', len(self.df)])
        writer.writerow(['Total Customers', self.df['Customer_ID'].nunique()])
        writer.writerow(['Total Products', self.df['Product_Name'].nunique()])
        writer.writerow(['Total Orders', self.df['Order_ID'].nunique()])

        if 'Order_Date' in self.df.columns:
            writer.writerow(['Date Range',
                             f"{self.df['Order_Date'].min()} to {self.df['Order_Date'].max()}"])

        writer.writerow([])

        # Revenue KPIs
        writer.writerow(['REVENUE KPIs'])
        writer.writerow(['KPI', 'Value'])
        total_sales = self.df['Sales'].sum()
        total_profit = self.df['Profit'].sum()
        writer.writerow(['Total Revenue', f'${total_sales:,.2f}'])
        writer.writerow(['Total Profit', f'${total_profit:,.2f}'])
        writer.writerow(['Profit Margin', f'{(total_profit/total_sales*100):.2f}%'])
        writer.writerow(['Average Order Value', f'${self.df["Sales"].mean():,.2f}'])
        writer.writerow(['Total Units Sold', int(self.df['Quantity'].sum())])

        writer.writerow([])

        # Top Customers
        writer.writerow(['TOP 10 CUSTOMERS BY REVENUE'])
        writer.writerow(['Customer Name', 'Total Spent', 'Orders', 'Avg Order Value'])
        top_customers = self.df.groupby('Customer_Name').agg(
            Total_Spent=('Sales', 'sum'),
            Orders=('Order_ID', 'nunique'),
            Avg_Value=('Sales', 'mean')
        ).nlargest(10, 'Total_Spent').reset_index()

        for _, row in top_customers.iterrows():
            writer.writerow([
                row['Customer_Name'],
                f'${row["Total_Spent"]:,.2f}',
                row['Orders'],
                f'${row["Avg_Value"]:,.2f}'
            ])

        writer.writerow([])

        # Category Performance
        writer.writerow(['CATEGORY PERFORMANCE'])
        writer.writerow(['Category', 'Sales', 'Profit', 'Margin', 'Units Sold'])
        category_perf = self.df.groupby('Category').agg(
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum'),
            Margin=('Profit_Margin_Pct', 'mean'),
            Units=('Quantity', 'sum')
        ).reset_index()

        for _, row in category_perf.iterrows():
            writer.writerow([
                row['Category'],
                f'${row["Sales"]:,.2f}',
                f'${row["Profit"]:,.2f}',
                f'{row["Margin"]:.2f}%',
                int(row['Units'])
            ])

        writer.writerow([])

        # Regional Performance
        writer.writerow(['REGIONAL PERFORMANCE'])
        writer.writerow(['Region', 'Sales', 'Profit', 'Orders', 'Customers'])
        region_perf = self.df.groupby('Region').agg(
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum'),
            Orders=('Order_ID', 'nunique'),
            Customers=('Customer_ID', 'nunique')
        ).reset_index()

        for _, row in region_perf.iterrows():
            writer.writerow([
                row['Region'],
                f'${row["Sales"]:,.2f}',
                f'${row["Profit"]:,.2f}',
                row['Orders'],
                row['Customers']
            ])

        writer.writerow([])

        # Monthly Trends
        writer.writerow(['MONTHLY SALES TRENDS'])
        writer.writerow(['Month', 'Sales', 'Profit', 'Orders'])
        if 'Year_Month' in self.df.columns:
            monthly = self.df.groupby('Year_Month').agg(
                Sales=('Sales', 'sum'),
                Profit=('Profit', 'sum'),
                Orders=('Order_ID', 'nunique')
            ).reset_index().sort_values('Year_Month')

            for _, row in monthly.iterrows():
                writer.writerow([
                    row['Year_Month'],
                    f'${row["Sales"]:,.2f}',
                    f'${row["Profit"]:,.2f}',
                    row['Orders']
                ])

        return output.getvalue()

    def generate_csv_export(self) -> str:
        """Generate full data export as CSV string."""
        self.logger.info("Generating full data CSV export...")
        return self.df.to_csv(index=False)

    def save_csv_report(self) -> Path:
        """Save CSV report to disk."""
        csv_content = self.generate_csv_summary()
        filename = f'sales_report_{self.report_date}.csv'
        path = REPORTS_DIR / filename

        with open(path, 'w', newline='') as f:
            f.write(csv_content)

        self.logger.info(f"CSV report saved to {path}")
        return path

    def save_full_data_export(self) -> Path:
        """Save full dataset as CSV export."""
        csv_content = self.generate_csv_export()
        filename = f'sales_data_export_{self.report_date}.csv'
        path = REPORTS_DIR / filename

        with open(path, 'w', newline='') as f:
            f.write(csv_content)

        self.logger.info(f"Full data export saved to {path}")
        return path

    def generate_pdf_report(self) -> bytes:
        """
        Generate a professional PDF report with charts and tables.
        Uses matplotlib to create visualizations embedded in the report.
        Returns PDF as bytes.
        """
        self.logger.info("Generating PDF report...")

        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            # Set style (fallback if seaborn-v0_8 style unavailable)
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except Exception:
                try:
                    plt.style.use('seaborn-darkgrid')
                except Exception:
                    plt.style.use('ggplot')
            plt.rcParams['figure.figsize'] = (11, 8.5)
            plt.rcParams['font.size'] = 10

            buffer = io.BytesIO()

            with PdfPages(buffer) as pdf:
                # Page 1: Cover Page
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')

                # Cover content
                ax.text(0.5, 0.85, 'AI-Powered Sales Analytics Dashboard',
                        fontsize=28, fontweight='bold', ha='center',
                        color='#1a1a2e', transform=ax.transAxes)
                ax.text(0.5, 0.75, 'Business Intelligence Report',
                        fontsize=18, ha='center', color='#16213e',
                        transform=ax.transAxes)
                ax.text(0.5, 0.60, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}',
                        fontsize=12, ha='center', color='#555',
                        transform=ax.transAxes)

                # Key metrics
                metrics_text = (
                    f"Total Revenue: ${self.df['Sales'].sum():,.2f}\n"
                    f"Total Profit: ${self.df['Profit'].sum():,.2f}\n"
                    f"Total Orders: {self.df['Order_ID'].nunique():,}\n"
                    f"Total Customers: {self.df['Customer_ID'].nunique():,}\n"
                    f"Data Period: {self.df['Order_Date'].min().strftime('%Y-%m-%d')} to "
                    f"{self.df['Order_Date'].max().strftime('%Y-%m-%d')}"
                )
                ax.text(0.5, 0.35, metrics_text, fontsize=14, ha='center',
                        color='#333', transform=ax.transAxes,
                        linespacing=1.8,
                        bbox=dict(boxstyle='round', facecolor='#f0f0f0',
                                  edgecolor='#ccc', pad=0.8))

                pdf.savefig(fig)
                plt.close()

                # Page 2: Sales Trends
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle('Sales Performance Overview', fontsize=18, fontweight='bold', y=0.98)

                # Monthly sales
                monthly = self.df.groupby('Year_Month').agg(
                    Sales=('Sales', 'sum'),
                    Profit=('Profit', 'sum')
                ).reset_index().sort_values('Year_Month')

                if len(monthly) > 0:
                    ax = axes[0, 0]
                    ax.plot(range(len(monthly)), monthly['Sales'] / 1000,
                            marker='o', linewidth=2, color='#2e86ab')
                    ax.set_title('Monthly Sales Trend', fontsize=12)
                    ax.set_ylabel('Sales ($K)')
                    ax.set_xlabel('Month')
                    ax.tick_params(axis='x', rotation=45)
                    n = max(1, len(monthly) // 8)
                    ax.set_xticks(range(0, len(monthly), n))
                    ax.set_xticklabels([monthly['Year_Month'].iloc[i] for i in range(0, len(monthly), n)],
                                       fontsize=7)

                    # Monthly profit
                    ax = axes[0, 1]
                    ax.bar(range(len(monthly)), monthly['Profit'] / 1000,
                           color=['#4e79a7' if p > 0 else '#e15759' for p in monthly['Profit']])
                    ax.set_title('Monthly Profit', fontsize=12)
                    ax.set_ylabel('Profit ($K)')
                    ax.axhline(y=0, color='black', linewidth=0.5)
                    ax.set_xticks(range(0, len(monthly), n))
                    ax.set_xticklabels([monthly['Year_Month'].iloc[i] for i in range(0, len(monthly), n)],
                                       fontsize=7)
                    ax.tick_params(axis='x', rotation=45)

                # Category distribution
                ax = axes[1, 0]
                category_sales = self.df.groupby('Category')['Sales'].sum()
                colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
                wedges, texts, autotexts = ax.pie(
                    category_sales.values,
                    labels=category_sales.index,
                    autopct='%1.1f%%',
                    colors=colors[:len(category_sales)],
                    startangle=90,
                    explode=[0.03] * len(category_sales)
                )
                ax.set_title('Sales by Category', fontsize=12)

                # Regional performance
                ax = axes[1, 1]
                region_sales = self.df.groupby('Region')['Sales'].sum().sort_values()
                ax.barh(range(len(region_sales)), region_sales.values / 1000,
                        color=['#4e79a7', '#f28e2b', '#e15759', '#76b7b2'][:len(region_sales)])
                ax.set_yticks(range(len(region_sales)))
                ax.set_yticklabels(region_sales.index)
                ax.set_xlabel('Sales ($K)')
                ax.set_title('Sales by Region', fontsize=12)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

                # Page 3: Customer Analysis
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
                fig.suptitle('Customer Analytics', fontsize=18, fontweight='bold', y=0.98)

                # Top customers
                if 'Customer_Segment' in self.df.columns:
                    ax = axes[0, 0]
                    segment_counts = self.df['Customer_Segment'].value_counts()
                    seg_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
                    segment_counts = segment_counts.reindex(
                        [s for s in seg_order if s in segment_counts.index]
                    )
                    colors = ['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2']
                    ax.bar(range(len(segment_counts)), segment_counts.values,
                           color=colors[:len(segment_counts)])
                    ax.set_xticks(range(len(segment_counts)))
                    ax.set_xticklabels(segment_counts.index)
                    ax.set_title('Customer Segments', fontsize=12)
                    ax.set_ylabel('Number of Customers')

                # Segment revenue
                ax = axes[0, 1]
                segment_revenue = self.df.groupby('Customer_Segment')['Sales'].sum()
                seg_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
                segment_revenue = segment_revenue.reindex(
                    [s for s in seg_order if s in segment_revenue.index]
                )
                colors = ['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2']
                ax.bar(range(len(segment_revenue)), segment_revenue.values / 1000,
                       color=colors[:len(segment_revenue)])
                ax.set_xticks(range(len(segment_revenue)))
                ax.set_xticklabels(segment_revenue.index)
                ax.set_title('Revenue by Customer Segment', fontsize=12)
                ax.set_ylabel('Revenue ($K)')

                # Top products
                top_products = self.df.groupby('Product_Name')['Sales'].sum().nlargest(10)
                ax = axes[1, 0]
                ax.barh(range(len(top_products)), top_products.values / 1000,
                        color='#2e86ab')
                ax.set_yticks(range(len(top_products)))
                ax.set_yticklabels([p[:25] + '...' if len(p) > 25 else p for p in top_products.index])
                ax.set_xlabel('Sales ($K)')
                ax.set_title('Top 10 Products', fontsize=12)

                # Discount analysis
                if 'Discount' in self.df.columns:
                    ax = axes[1, 1]
                    discount_bins = pd.cut(self.df['Discount'],
                                           bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0])
                    discount_analysis = self.df.groupby(discount_bins).agg(
                        Sales=('Sales', 'sum'),
                        Profit=('Profit', 'sum')
                    )
                    x = range(len(discount_analysis))
                    width = 0.35
                    ax.bar([i - width/2 for i in x],
                           discount_analysis['Sales'].values / 1000,
                           width, label='Sales ($K)', color='#4e79a7')
                    ax.bar([i + width/2 for i in x],
                           discount_analysis['Profit'].values / 1000,
                           width, label='Profit ($K)', color='#59a14f')
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(b) for b in discount_analysis.index], fontsize=8)
                    ax.set_title('Sales & Profit by Discount Range', fontsize=12)
                    ax.legend(fontsize=8)
                    ax.tick_params(axis='x', rotation=45)

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

                # Page 4: KPIs Summary Table
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis('off')
                ax.set_title('Key Performance Indicators', fontsize=18, fontweight='bold', pad=20)

                kpis = [
                    ('Revenue Metrics', [
                        ('Total Revenue', f'${self.df["Sales"].sum():,.2f}'),
                        ('Average Order Value', f'${self.df["Sales"].mean():,.2f}'),
                        ('Revenue per Customer', f'${self.df["Sales"].sum()/self.df["Customer_ID"].nunique():,.2f}'),
                        ('Revenue per Product', f'${self.df["Sales"].sum()/self.df["Product_Name"].nunique():,.2f}'),
                    ]),
                    ('Profit Metrics', [
                        ('Total Profit', f'${self.df["Profit"].sum():,.2f}'),
                        ('Profit Margin', f'{(self.df["Profit"].sum()/self.df["Sales"].sum()*100):.2f}%'),
                        ('Profit per Order', f'${self.df["Profit"].sum()/self.df["Order_ID"].nunique():,.2f}'),
                    ]),
                    ('Operational Metrics', [
                        ('Total Orders', f'{self.df["Order_ID"].nunique():,}'),
                        ('Total Customers', f'{self.df["Customer_ID"].nunique():,}'),
                        ('Total Products', f'{self.df["Product_Name"].nunique():,}'),
                        ('Total Units Sold', f'{int(self.df["Quantity"].sum()):,}'),
                        ('Items per Order', f'{self.df["Quantity"].sum()/self.df["Order_ID"].nunique():.1f}'),
                    ]),
                    ('Discount Metrics', [
                        ('Average Discount', f'{self.df["Discount"].mean()*100:.1f}%'),
                        ('Total Discount Impact', f'${(self.df["Sales"]*self.df["Discount"]).sum():,.2f}'),
                        ('Orders with Discount', f'{(self.df["Discount"]>0).sum():,}'),
                    ]),
                ]

                y_pos = 0.85
                for section_name, metrics in kpis:
                    ax.text(0.1, y_pos, section_name, fontsize=13, fontweight='bold',
                            color='#1a1a2e', transform=ax.transAxes)
                    y_pos -= 0.05

                    for metric_name, metric_value in metrics:
                        ax.text(0.15, y_pos, metric_name, fontsize=10,
                                color='#333', transform=ax.transAxes)
                        ax.text(0.55, y_pos, metric_value, fontsize=10,
                                fontweight='bold', color='#2e86ab',
                                transform=ax.transAxes)
                        y_pos -= 0.035

                    y_pos -= 0.02

                # Footer
                ax.text(0.5, 0.02,
                        'Generated by AI-Powered Sales Analytics Dashboard',
                        fontsize=9, ha='center', color='#888',
                        transform=ax.transAxes)

                pdf.savefig(fig)
                plt.close()

            pdf_bytes = buffer.getvalue()
            buffer.close()

            self.logger.info(f"PDF report generated ({len(pdf_bytes):,} bytes)")
            return pdf_bytes

        except ImportError as e:
            self.logger.error(f"PDF generation failed - missing dependency: {e}")
            self.logger.error("Install matplotlib and reportlab: pip install matplotlib reportlab")
            return b''

    def save_pdf_report(self) -> Path:
        """Save PDF report to disk."""
        pdf_bytes = self.generate_pdf_report()
        if pdf_bytes:
            filename = f'sales_report_{self.report_date}.pdf'
            path = REPORTS_DIR / filename
            with open(path, 'wb') as f:
                f.write(pdf_bytes)
            self.logger.info(f"PDF report saved to {path}")
            return path
        return None

    def generate_insights_summary(self) -> str:
        """Generate a text summary of key business insights."""
        self.logger.info("Generating insights summary...")

        total_sales = self.df['Sales'].sum()
        total_profit = self.df['Profit'].sum()
        margin = (total_profit / total_sales * 100) if total_sales > 0 else 0

        insights = []
        insights.append("=" * 60)
        insights.append("AI-POWERED BUSINESS INSIGHTS SUMMARY")
        insights.append("=" * 60)
        insights.append("")

        # Executive Summary
        insights.append("EXECUTIVE SUMMARY")
        insights.append("-" * 40)
        insights.append(
            f"Total revenue of ${total_sales:,.2f} with ${total_profit:,.2f} "
            f"in profit ({margin:.1f}% margin) from "
            f"{self.df['Order_ID'].nunique():,} orders across "
            f"{self.df['Customer_ID'].nunique():,} customers."
        )
        insights.append("")

        # Key Findings
        insights.append("KEY FINDINGS")
        insights.append("-" * 40)

        # Best category
        best_category = self.df.groupby('Category')['Sales'].sum().idxmax()
        best_cat_sales = self.df.groupby('Category')['Sales'].sum().max()
        insights.append(f"• Top Category: '{best_category}' with ${best_cat_sales:,.2f} in sales")

        # Best region
        best_region = self.df.groupby('Region')['Sales'].sum().idxmax()
        best_reg_sales = self.df.groupby('Region')['Sales'].sum().max()
        insights.append(f"• Top Region: '{best_region}' with ${best_reg_sales:,.2f} in sales")

        # Customer concentration
        top_10_pct = self.df.groupby('Customer_ID')['Sales'].sum().nlargest(
            int(self.df['Customer_ID'].nunique() * 0.1)
        ).sum()
        top_10_share = (top_10_pct / total_sales * 100)
        insights.append(f"• Top 10% of customers contribute {top_10_share:.1f}% of revenue")

        # Average metrics
        avg_aov = self.df['Sales'].mean()
        avg_qty = self.df['Quantity'].mean()
        insights.append(f"• Average order value: ${avg_aov:,.2f} ({avg_qty:.1f} items per order)")

        insights.append("")

        # Recommendations
        insights.append("ACTIONABLE RECOMMENDATIONS")
        insights.append("-" * 40)
        insights.append("1. Focus marketing spend on top-performing categories and regions")
        insights.append("2. Implement loyalty programs for high-value customer segments")
        insights.append("3. Optimize discount strategy to protect profit margins")
        insights.append("4. Explore expansion opportunities in underperforming regions")
        insights.append("5. Leverage seasonal trends for promotional campaigns")
        insights.append("")

        insights.append("=" * 60)
        insights.append("End of Report")
        insights.append("=" * 60)

        return '\n'.join(insights)

    def save_insights_report(self) -> Path:
        """Save insights summary to disk."""
        content = self.generate_insights_summary()
        filename = f'business_insights_{self.report_date}.txt'
        path = REPORTS_DIR / filename

        with open(path, 'w') as f:
            f.write(content)

        self.logger.info(f"Insights report saved to {path}")
        return path

    def generate_anomaly_report(self, anomalies: pd.DataFrame) -> str:
        """Generate a report on detected anomalies."""
        self.logger.info("Generating anomaly detection report...")

        if anomalies.empty:
            return "No anomalies detected in the dataset."

        output = []
        output.append("=" * 60)
        output.append("ANOMALY DETECTION REPORT")
        output.append("=" * 60)
        output.append("")
        output.append(f"Total anomalies detected: {len(anomalies)}")
        output.append("")

        if 'Anomaly_Type' in anomalies.columns:
            output.append("ANOMALY BREAKDOWN:")
            for atype, count in anomalies['Anomaly_Type'].value_counts().items():
                output.append(f"  {atype}: {count}")
            output.append("")

        output.append("TOP ANOMALIES:")
        display_cols = [c for c in ['Order_Date', 'Sales', 'Orders', 'Anomaly_Type']
                       if c in anomalies.columns]
        top_anomalies = anomalies.head(10)[display_cols]

        for _, row in top_anomalies.iterrows():
            output.append(f"  {row.to_dict()}")

        return '\n'.join(output)

    def generate_all_reports(self, anomalies: pd.DataFrame = None) -> dict:
        """Generate all reports."""
        self.logger.info("=" * 60)
        self.logger.info("GENERATING ALL REPORTS")
        self.logger.info("=" * 60)

        reports = {
            'csv_report_path': self.save_csv_report(),
            'full_data_export_path': self.save_full_data_export(),
            'insights_report_path': self.save_insights_report(),
            'pdf_report_path': self.save_pdf_report(),
        }

        if anomalies is not None:
            anomaly_text = self.generate_anomaly_report(anomalies)
            anomaly_path = REPORTS_DIR / f'anomaly_report_{self.report_date}.txt'
            with open(anomaly_path, 'w') as f:
                f.write(anomaly_text)
            reports['anomaly_report_path'] = anomaly_path

        self.logger.info("\nAll reports generated!")
        return reports


if __name__ == '__main__':
    from data_loader import load_dataset
    from preprocessing import run_preprocessing_pipeline
    from analysis import SalesAnalyzer

    df = load_dataset()
    df = run_preprocessing_pipeline(df)

    analyzer = SalesAnalyzer(df)
    analysis = analyzer.run_all_analyses()

    report_gen = ReportGenerator(df, analysis)
    reports = report_gen.generate_all_reports()

    print("\n=== REPORTS GENERATED ===")
    for name, path in reports.items():
        print(f"  {name}: {path}")
