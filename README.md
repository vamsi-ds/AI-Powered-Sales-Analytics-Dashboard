# 📊 AI-Powered Sales Analytics Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-F7931E)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-00BFFF)](https://xgboost.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15%2B-3B82F6)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **A production-quality, recruiter-ready Data Analytics & Data Science portfolio project.**  
> An end-to-end business intelligence solution featuring automated data pipelines, machine learning forecasting, interactive dashboards, and AI-powered business insights.

---

## ✨ Features

### 📈 Comprehensive Sales Analytics
- Sales trend analysis (daily, monthly, quarterly, yearly)
- Customer behavior analysis & segmentation (RFM analysis)
- Regional & geographic performance analysis
- Product & category performance analysis
- Profitability & discount impact analysis
- Year-over-Year growth metrics

### 🤖 Machine Learning Forecasting
- **Prophet Model**: Facebook's time series forecasting with seasonality
- **XGBoost Model**: Gradient boosting with engineered features
- Trend decomposition & seasonality analysis
- Confidence intervals & accuracy metrics
- Anomaly detection (IQR, Z-score, Isolation Forest)

### 📊 Interactive Dashboard
- Modern dark theme with professional design
- Real-time KPI cards with performance metrics
- Interactive charts (Plotly) with hover details
- Multi-page navigation & filters
- Responsive layout for all screen sizes

### 💾 Data Pipeline
- Automated data loading (Kaggle, CSV, or synthetic generation)
- Data cleaning & validation
- Feature engineering (30+ engineered features)
- SQLite database with optimized queries
- Comprehensive preprocessing pipeline

### 📋 Reporting
- PDF report generation with embedded charts
- CSV summary reports
- Full data export
- AI-generated business insights
- Anomaly detection reports

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/sales-analytics-dashboard.git
cd sales-analytics-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete pipeline
python main.py

# 4. Launch the interactive dashboard
python main.py --dashboard
```

The dashboard will be available at **http://localhost:8501**

---

## 🎯 Usage Guide

### Running the Pipeline

```bash
# Run the complete data pipeline (load, clean, analyze, forecast, report)
python main.py

# Launch the interactive Streamlit dashboard
python main.py --dashboard

# Generate reports only
python main.py --report

# Load and process data only
python main.py --data-only

# Get help
python main.py --help
```

### Dashboard Navigation

| Tab | Description |
|-----|-------------|
| **📈 Overview** | High-level KPIs and performance metrics |
| **📊 Sales Analytics** | Detailed sales trends and time analysis |
| **👥 Customer Insights** | Customer segmentation and behavior analysis |
| **🌍 Regional Analysis** | Geographic sales performance |
| **📦 Product Analysis** | Product and category performance |
| **🤖 ML Forecasting** | Machine learning sales predictions |
| **💡 Business Insights** | AI-powered recommendations |
| **📋 Reports** | Export and reporting options |

### Filters

Use the sidebar filters to:
- Filter by **Year** (multi-select)
- Filter by **Category** (Technology, Furniture, Office Supplies)
- Filter by **Region** (East, West, Central, South)
- Filter by **Customer Segment** (Bronze, Silver, Gold, Platinum)

---

## 📁 Project Structure

```
📦 sales-analytics-dashboard
├── 📂 data/
│   ├── 📂 raw/              # Raw datasets
│   └── 📂 processed/        # Cleaned & processed data
├── 📂 dashboard/
│   └── 🐍 app.py            # Streamlit dashboard application
├── 📂 models/               # Saved ML models & forecasts
├── 📂 notebooks/            # Jupyter notebooks (optional)
├── 📂 reports/              # Generated reports (PDF, CSV, TXT)
├── 📂 assets/               # Static assets
├── 🐍 main.py               # Main entry point & pipeline orchestrator
├── 🐍 data_loader.py        # Dataset loading & generation
├── 🐍 preprocessing.py      # Data cleaning & feature engineering
├── 🐍 analysis.py           # Comprehensive data analysis
├── 🐍 forecasting.py        # ML forecasting (Prophet/XGBoost)
├── 🐍 sql_engine.py         # SQLite analytics engine
├── 🐍 report_generator.py   # PDF/CSV report generation
├── 🗄️ database.db           # SQLite analytics database
├── 📄 requirements.txt      # Python dependencies
└── 📄 README.md             # Project documentation
```

---

## 🔧 Technical Architecture

### Data Pipeline
```
Raw Data (CSV/Kaggle/Synthetic)
    ↓
Data Loader (data_loader.py)
    ↓
Preprocessing & Feature Engineering (preprocessing.py)
    ├── Missing value handling
    ├── Duplicate removal
    ├── Outlier treatment
    └── Feature engineering (30+ features)
    ↓
SQLite Database (sql_engine.py)
    ├── Optimized indexes
    └── Business intelligence queries
    ↓
Analysis Engine (analysis.py)
    ├── Sales trends
    ├── Customer analysis
    ├── Regional analysis
    ├── Product analysis
    └── KPI generation
    ↓
ML Forecasting (forecasting.py)
    ├── Prophet model
    ├── XGBoost model
    └── Anomaly detection
    ↓
Report Generation (report_generator.py)
    ├── PDF reports
    ├── CSV exports
    └── Business insights
    ↓
Interactive Dashboard (dashboard/app.py)
```

### Machine Learning Models

| Model | Use Case | Features |
|-------|----------|----------|
| **Prophet** | Time series forecasting | Seasonality, trend, holidays |
| **XGBoost** | Sales prediction | Engineered features, lags, rolling stats |
| **IQR/Z-score** | Anomaly detection | Statistical outlier detection |
| **Isolation Forest** | Anomaly detection | ML-based anomaly detection |

---

## 📊 Dataset

The project automatically handles data acquisition:

1. **Kaggle Datasets**: Attempts to download from popular sales datasets
2. **CSV Sources**: Falls back to public CSV repositories
3. **Synthetic Generation**: Automatically generates realistic enterprise sales data

The generated dataset includes:
- 5,000+ sales records
- 500+ customers
- 200+ products across 3 categories
- 5-year date range (2019-2024)
- 15+ attributes per record
- Realistic pricing and seasonal patterns

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎯 Portfolio Value

This project is designed to showcase:

### For Data Analyst Roles
- **SQL proficiency**: Complex queries, aggregations, window functions
- **Data visualization**: Interactive dashboards, chart design
- **Business acumen**: KPI definition, trend analysis, actionable insights
- **Reporting**: Automated report generation, data storytelling

### For Data Science Roles
- **Machine Learning**: Time series forecasting, feature engineering
- **Statistical analysis**: Trend decomposition, anomaly detection
- **Python engineering**: Modular code, OOP, logging, error handling
- **Data pipeline**: ETL processes, data cleaning, validation

### For BI Analyst Roles
- **Dashboard design**: User experience, visual hierarchy, dark theme
- **Business metrics**: Revenue analysis, customer segmentation, profitability
- **Interactive filtering**: Real-time data exploration
- **Automated insights**: AI-powered recommendations

---

## 📞 Contact

**Your Name** - [@your_twitter](https://twitter.com/your_twitter) - email@example.com

Project Link: [https://github.com/yourusername/sales-analytics-dashboard](https://github.com/yourusername/sales-analytics-dashboard)

---

<div align="center">
⭐ **If you found this project useful, please give it a star!** ⭐
</div>
