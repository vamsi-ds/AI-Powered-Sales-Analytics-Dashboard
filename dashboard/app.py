"""
AI-Powered Sales Analytics Dashboard
Streamlit-based interactive dashboard with dark theme, KPI cards, and visualizations.
"""

import sys
import os
import logging
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_dataset
from preprocessing import run_preprocessing_pipeline
from analysis import SalesAnalyzer
from forecasting import SalesForecaster
from sql_engine import SQLAnalyticsEngine
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="AI Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS - Dark Theme with Modern Design
# =============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: #0f1117;
        color: #e0e0e0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16181f 0%, #1a1d27 100%);
        border-right: 1px solid #2a2d3a;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #b0b8d1;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #2e3145;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.4);
    }
    div[data-testid="metric-container"] label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e0e0e0 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #64ffda !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    h1 {
        font-size: 2rem !important;
        background: linear-gradient(135deg, #64ffda, #48bfe3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0 !important;
    }

    /* Subheader */
    .subheader-text {
        color: #8892b0;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #16181f;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2a2d3a;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 20px !important;
        color: #8892b0;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e86ab, #48bfe3) !important;
        color: #ffffff !important;
    }

    /* DataFrames */
    .dataframe-container {
        background: #1e2130;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #2e3145;
    }
    .stDataFrame {
        background: transparent !important;
    }
    .stDataFrame [data-testid="StyledDataFrameColHeader"] {
        background: #252840 !important;
        color: #64ffda !important;
    }
    .stDataFrame td {
        color: #e0e0e0 !important;
    }

    /* Select boxes and inputs */
    .stSelectbox label, .stMultiSelect label {
        color: #8892b0 !important;
        font-weight: 500 !important;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background: #1e2130 !important;
        border-color: #2e3145 !important;
        color: #e0e0e0 !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #2e86ab, #48bfe3) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 24px !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(46, 134, 171, 0.4) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1e2130 !important;
        border-radius: 8px !important;
        border: 1px solid #2e3145 !important;
        color: #e0e0e0 !important;
    }

    /* Plots */
    .js-plotly-plot {
        border-radius: 12px !important;
        background: #1e2130 !important;
        padding: 8px;
        border: 1px solid #2e3145 !important;
    }

    /* Info boxes */
    .stAlert {
        background: #1e2130 !important;
        border: 1px solid #2e3145 !important;
        border-radius: 12px !important;
        border-left: 4px solid #64ffda !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2e86ab, #64ffda) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #16181f;
    }
    ::-webkit-scrollbar-thumb {
        background: #2e3145;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3e4165;
    }

    /* Cards for insights */
    .insight-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2e3145;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: transform 0.2s ease;
    }
    .insight-card:hover {
        transform: translateX(4px);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #5a5d7a;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid #2a2d3a;
        margin-top: 3rem;
    }

    /* Fix white text on white bg for plotly */
    .stPlotlyChart {
        background: transparent !important;
    }
    .stPlotlyChart > div {
        background: transparent !important;
    }

    /* Remove padding from main container */
    .main > div {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PLOTLY THEME CONFIG
# =============================================================================
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#e0e0e0', 'family': 'Inter, sans-serif'},
        'title': {'font': {'color': '#e0e0e0', 'size': 18}},
        'xaxis': {
            'gridcolor': '#2a2d3a',
            'zerolinecolor': '#2a2d3a',
            'tickfont': {'color': '#8892b0'},
            'title': {'font': {'color': '#8892b0'}}
        },
        'yaxis': {
            'gridcolor': '#2a2d3a',
            'zerolinecolor': '#2a2d3a',
            'tickfont': {'color': '#8892b0'},
            'title': {'font': {'color': '#8892b0'}}
        },
        'legend': {
            'font': {'color': '#e0e0e0'},
            'bgcolor': 'rgba(0,0,0,0)'
        },
        'hoverlabel': {
            'bgcolor': '#1e2130',
            'font': {'color': '#e0e0e0', 'size': 12}
        },
        'colorway': ['#2e86ab', '#48bfe3', '#64ffda', '#f28e2b', '#e15759',
                      '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']
    }
}

# Color palette for charts
COLORS = ['#2e86ab', '#48bfe3', '#64ffda', '#f28e2b', '#e15759',
          '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7']

COLOR_MAP = {
    'Technology': '#2e86ab',
    'Furniture': '#f28e2b',
    'Office Supplies': '#59a14f'
}

# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================
@st.cache_data(show_spinner="📊 Loading sales dataset...")
def load_sales_data():
    """Load and preprocess sales data."""
    with st.spinner("Loading and processing data..."):
        raw_df = load_dataset()
        df = run_preprocessing_pipeline(raw_df)
        return df

@st.cache_data(show_spinner="🔍 Running analysis...")
def run_analysis(df):
    """Run comprehensive analysis."""
    analyzer = SalesAnalyzer(df)
    return analyzer.run_all_analyses()

@st.cache_resource(show_spinner="💾 Initializing SQL engine...")
def init_sql_engine(df):
    """Initialize SQL analytics engine."""
    engine = SQLAnalyticsEngine()
    engine.load_data(df)
    return engine

@st.cache_data(show_spinner="🤖 Running ML forecasting...")
def run_forecasting(df):
    """Run forecasting models."""
    forecaster = SalesForecaster(df)
    results = forecaster.run_all_forecasts()
    return results

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_currency(value):
    """Format currency values."""
    if abs(value) >= 1_000_000:
        return f'${value/1_000_000:.2f}M'
    elif abs(value) >= 1_000:
        return f'${value/1_000:.1f}K'
    else:
        return f'${value:,.2f}'

def format_percent(value):
    """Format percentage values."""
    return f'{value:.1f}%'

def apply_plotly_theme(fig):
    """Apply custom dark theme to plotly figure."""
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    fig.update_xaxes(
        gridcolor='#2a2d3a',
        zerolinecolor='#2a2d3a',
        tickfont={'color': '#8892b0'}
    )
    fig.update_yaxes(
        gridcolor='#2a2d3a',
        zerolinecolor='#2a2d3a',
        tickfont={'color': '#8892b0'}
    )
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    """Render the sidebar with navigation and filters."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <h2 style="margin: 0; font-size: 1.3rem; background: linear-gradient(135deg, #64ffda, #48bfe3);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Sales Analytics</h2>
            <p style="color: #8892b0; font-size: 0.75rem; margin-top: 0.3rem;">
                AI-Powered Dashboard v2.0
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown("### 🧭 Navigation")
        nav_options = [
            "📈 Overview",
            "📊 Sales Analytics",
            "👥 Customer Insights",
            "🌍 Regional Analysis",
            "📦 Product Analysis",
            "🤖 ML Forecasting",
            "💡 Business Insights",
            "📋 Reports"
        ]

        selected = st.radio(
            "Navigate",
            nav_options,
            label_visibility="collapsed",
            index=0
        )

        st.divider()

        # Filters
        st.markdown("### 🎯 Filters")

        if 'df' in st.session_state:
            df = st.session_state.df

            # Year filter
            years = sorted(df['Year'].unique())
            selected_years = st.multiselect(
                "Year",
                years,
                default=years,
                help="Filter by year"
            )

            # Category filter
            categories = ['All'] + sorted(df['Category'].unique())
            selected_category = st.selectbox(
                "Category",
                categories,
                help="Filter by product category"
            )

            # Region filter
            regions = ['All'] + sorted(df['Region'].unique())
            selected_region = st.selectbox(
                "Region",
                regions,
                help="Filter by region"
            )

            # Customer segment filter
            if 'Customer_Segment' in df.columns:
                segments = ['All'] + sorted(df['Customer_Segment'].unique())
                selected_segment = st.selectbox(
                    "Customer Segment",
                    segments,
                    help="Filter by customer segment"
                )
            else:
                selected_segment = 'All'

            # Apply filters
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]
            if selected_category != 'All':
                filtered_df = filtered_df[filtered_df['Category'] == selected_category]
            if selected_region != 'All':
                filtered_df = filtered_df[filtered_df['Region'] == selected_region]
            if selected_segment != 'All':
                filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]

            st.session_state.filtered_df = filtered_df

        st.divider()

        # Action buttons
        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        if st.button("📥 Export Report", use_container_width=True):
            st.session_state.show_export_options = True

        st.divider()

        # Status
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; color: #5a5d7a; font-size: 0.75rem;">
            <div>🟢 System Online</div>
            <div style="margin-top: 4px;">Data updated continuously</div>
        </div>
        """, unsafe_allow_html=True)

    return selected

# =============================================================================
# DASHBOARD SECTIONS
# =============================================================================
def render_overview():
    """Render the main overview page with KPIs."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis
    kpis = analysis['kpis']

    # Title
    st.markdown("<h1>📈 Sales Analytics Overview</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Real-time business intelligence and performance metrics</p>',
                unsafe_allow_html=True)

    # KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=format_currency(kpis['Total_Revenue']),
            delta=f"{kpis['Total_Orders']:,} Orders"
        )

    with col2:
        st.metric(
            label="📈 Total Profit",
            value=format_currency(kpis['Total_Profit']),
            delta=f"{kpis['Profit_Margin']:.1f}% Margin",
            delta_color="normal"
        )

    with col3:
        st.metric(
            label="👥 Total Customers",
            value=f"{kpis['Total_Customers']:,}",
            delta=f"${kpis['Revenue_per_Customer']:,.0f} avg"
        )

    with col4:
        st.metric(
            label="📦 Avg Order Value",
            value=format_currency(kpis['Average_Order_Value']),
            delta=f"{kpis['Items_per_Order']:.1f} items/order"
        )

    # Second row of KPIs
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Products",
            value=f"{kpis['Total_Products']:,}",
            delta=f"{kpis['Products_per_Order']:.1f} per order"
        )

    with col2:
        st.metric(
            label="🏷️ Avg Discount",
            value=f"{kpis['Average_Discount']*100:.1f}%",
            delta=f"Impact: {format_currency(kpis['Total_Discount_Impact'])}"
        )

    with col3:
        st.metric(
            label="📋 Total Quantity Sold",
            value=f"{kpis['Total_Quantity_Sold']:,}",
            delta=f"{kpis['Items_per_Order']:.1f} avg per order"
        )

    with col4:
        st.metric(
            label="📅 Date Range",
            value=f"{kpis['Date_Start'].strftime('%b %Y')} - {kpis['Date_End'].strftime('%b %Y')}",
            delta=f"{(kpis['Date_End'] - kpis['Date_Start']).days} days"
        )

    st.divider()

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Monthly Sales Trend")
        monthly_trends = analysis['sales_trends']['monthly_trends']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_trends['Year_Month'],
            y=monthly_trends['Total_Sales'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#2e86ab', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        fig.add_trace(go.Bar(
            x=monthly_trends['Year_Month'],
            y=monthly_trends['Total_Profit'],
            name='Profit',
            marker_color='#64ffda',
            opacity=0.6,
            yaxis='y2'
        ))
        fig.update_layout(
            height=350,
            yaxis2=dict(
                overlaying='y',
                side='right',
                title='Profit ($)',
                gridcolor='#2a2d3a'
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Top Products by Revenue")
        top_products = analysis['product_analysis']['top_products'].head(10)

        fig = go.Figure(go.Bar(
            x=top_products['Total_Sales'],
            y=top_products['Product_Name'],
            orientation='h',
            marker=dict(
                color=top_products['Total_Sales'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title='Revenue ($)', tickformat='$,.0f')
            ),
            text=top_products['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0')
        ))
        fig.update_layout(
            height=350,
            yaxis=dict(autorange='reversed'),
            xaxis=dict(showgrid=False)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Second row of charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 Sales by Category")
        category_stats = analysis['category_analysis']['category_stats']

        fig = go.Figure(data=[
            go.Pie(
                labels=category_stats['Category'],
                values=category_stats['Total_Sales'],
                hole=0.4,
                marker=dict(colors=[COLOR_MAP.get(c, '#2e86ab') for c in category_stats['Category']]),
                textinfo='label+percent',
                textfont=dict(color='#e0e0e0', size=12),
                hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>'
            )
        ])
        fig.update_layout(height=350, showlegend=False)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🌍 Regional Performance")
        region_perf = analysis['regional_analysis']['region_performance']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=region_perf['Region'],
            y=region_perf['Total_Sales'],
            name='Sales',
            marker_color='#2e86ab',
            text=region_perf['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0')
        ))
        fig.add_trace(go.Scatter(
            x=region_perf['Region'],
            y=region_perf['Total_Profit'],
            name='Profit',
            mode='lines+markers',
            line=dict(color='#64ffda', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(height=350, hovermode='x unified')
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_sales_analytics():
    """Render detailed sales analytics page."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis

    st.markdown("<h1>📊 Sales Analytics</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Deep dive into sales performance and trends</p>',
                unsafe_allow_html=True)

    # Time period selector
    period = st.selectbox(
        "Select Time Granularity",
        ["Monthly", "Quarterly", "Yearly", "Daily"],
        help="Choose the time aggregation level"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📈 {period} Sales Trend")

        if period == "Monthly":
            trends = analysis['sales_trends']['monthly_trends']
            x_col = 'Year_Month'
        elif period == "Quarterly":
            trends = analysis['sales_trends']['quarterly_trends']
            trends['Label'] = trends['Year'].astype(str) + '-Q' + trends['Quarter'].astype(str)
            x_col = 'Label'
        elif period == "Yearly":
            trends = analysis['sales_trends']['yearly_analysis']
            x_col = 'Year'
        else:
            trends = analysis['sales_trends']['daily_sales']
            x_col = 'Order_Date'

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trends[x_col],
            y=trends['Total_Sales'],
            mode='lines+markers',
            name='Sales',
            line=dict(color='#2e86ab', width=3),
            fill='tozeroy',
            fillcolor='rgba(46, 134, 171, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=trends[x_col],
            y=trends['Total_Profit'],
            mode='lines+markers',
            name='Profit',
            line=dict(color='#64ffda', width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            height=400,
            hovermode='x unified',
            yaxis2=dict(overlaying='y', side='right', gridcolor='#2a2d3a'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Sales & Profit Distribution")

        # Create a heatmap-like view with category vs month
        pivot_data = df.pivot_table(
            values='Sales',
            index='Category',
            columns='Year_Month',
            aggfunc='sum',
            fill_value=0
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Teal',
            hovertemplate='<b>%{y}</b><br>%{x}<br>Sales: $%{z:,.2f}<extra></extra>',
            colorbar=dict(title='Sales ($)', tickformat='$,.0f')
        ))
        fig.update_layout(
            height=400,
            xaxis=dict(tickangle=45, nticks=12)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Day of week & seasonal analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📅 Sales by Day of Week")
        dow = analysis['sales_trends']['day_of_week_analysis']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dow['Day_Name'],
            y=dow['Total_Sales'],
            marker=dict(
                color=dow['Total_Sales'],
                colorscale='Viridis',
                showscale=True
            ),
            text=dow['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0', size=10)
        ))
        fig.update_layout(height=350)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🌤️ Seasonal Analysis")
        seasonal = analysis['sales_trends']['seasonal_analysis']

        fig = go.Figure()
        colors = {'Winter': '#48bfe3', 'Spring': '#59a14f', 'Summer': '#f28e2b', 'Fall': '#e15759'}
        fig.add_trace(go.Bar(
            x=seasonal['Season'],
            y=seasonal['Total_Sales'],
            marker_color=[colors.get(s, '#2e86ab') for s in seasonal['Season']],
            text=seasonal['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0', size=10)
        ))
        fig.add_trace(go.Scatter(
            x=seasonal['Season'],
            y=seasonal['Total_Profit'],
            mode='lines+markers',
            name='Profit',
            line=dict(color='#64ffda', width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            height=350,
            yaxis2=dict(overlaying='y', side='right', gridcolor='#2a2d3a')
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_customer_insights():
    """Render customer insights page."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis

    st.markdown("<h1>👥 Customer Insights</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Customer behavior analysis and segmentation</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 Top Customers")

        top_customers = analysis['customer_analysis']['top_customers']

        fig = go.Figure(go.Bar(
            x=top_customers['Total_Spent'],
            y=top_customers['Customer_Name'].str[:20] + '...',
            orientation='h',
            marker=dict(
                color=top_customers['Total_Spent'],
                colorscale='Tealgrn',
                showscale=True,
                colorbar=dict(title='Revenue ($)')
            ),
            text=top_customers['Total_Spent'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0')
        ))
        fig.update_layout(height=400, yaxis=dict(autorange='reversed'))
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Customer Segments")
        segment_dist = analysis['customer_analysis']['segment_distribution']

        # Segment revenue
        segment_revenue = df.groupby('Customer_Segment').agg(
            Revenue=('Sales', 'sum'),
            Customers=('Customer_ID', 'nunique')
        ).reset_index()

        seg_order = ['Bronze', 'Silver', 'Gold', 'Platinum']
        segment_revenue['Order'] = segment_revenue['Customer_Segment'].map(
            {s: i for i, s in enumerate(seg_order)}
        )
        segment_revenue = segment_revenue.sort_values('Order')

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=['Customers', 'Revenue']
        )

        colors_seg = ['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2']

        fig.add_trace(go.Pie(
            labels=segment_revenue['Customer_Segment'],
            values=segment_revenue['Customers'],
            marker=dict(colors=colors_seg[:len(segment_revenue)]),
            hole=0.4,
            textinfo='label+percent'
        ), 1, 1)

        fig.add_trace(go.Pie(
            labels=segment_revenue['Customer_Segment'],
            values=segment_revenue['Revenue'],
            marker=dict(colors=colors_seg[:len(segment_revenue)]),
            hole=0.4,
            textinfo='label+percent'
        ), 1, 2)

        fig.update_layout(height=400, showlegend=False)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Customer segment analysis table
    st.markdown("### 📋 Customer Segment Analysis")
    if 'Customer_Segment' in df.columns:
        segment_analysis = df.groupby('Customer_Segment').agg(
            Customer_Count=('Customer_ID', 'nunique'),
            Total_Revenue=('Sales', 'sum'),
            Avg_Order_Value=('Sales', 'mean'),
            Total_Profit=('Profit', 'sum'),
            Avg_Profit_Margin=('Profit_Margin_Pct', 'mean'),
            Avg_Orders=('Customer_Total_Orders', 'mean')
        ).reset_index()

        segment_analysis['Total_Revenue'] = segment_analysis['Total_Revenue'].apply(
            lambda x: f'${x:,.2f}'
        )
        segment_analysis['Avg_Order_Value'] = segment_analysis['Avg_Order_Value'].apply(
            lambda x: f'${x:,.2f}'
        )
        segment_analysis['Total_Profit'] = segment_analysis['Total_Profit'].apply(
            lambda x: f'${x:,.2f}'
        )
        segment_analysis['Avg_Profit_Margin'] = segment_analysis['Avg_Profit_Margin'].apply(
            lambda x: f'{x:.1f}%'
        )
        segment_analysis['Avg_Orders'] = segment_analysis['Avg_Orders'].apply(
            lambda x: f'{x:.1f}'
        )

        st.dataframe(
            segment_analysis,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Customer_Segment': 'Segment',
                'Customer_Count': 'Customers',
                'Total_Revenue': 'Revenue',
                'Avg_Order_Value': 'Avg Order',
                'Total_Profit': 'Profit',
                'Avg_Profit_Margin': 'Margin',
                'Avg_Orders': 'Avg Orders'
            }
        )

    # Payment mode analysis
    st.markdown("### 💳 Payment Mode Analysis")
    payment_analysis = df.groupby('Payment_Mode').agg(
        Orders=('Order_ID', 'nunique'),
        Revenue=('Sales', 'sum'),
        Avg_Value=('Sales', 'mean')
    ).reset_index().sort_values('Revenue', ascending=False)

    fig = go.Figure(go.Pie(
        labels=payment_analysis['Payment_Mode'],
        values=payment_analysis['Revenue'],
        hole=0.4,
        marker=dict(colors=COLORS[:len(payment_analysis)]),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>'
    ))
    fig.update_layout(height=350)
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_regional_analysis():
    """Render regional analysis page."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis

    st.markdown("<h1>🌍 Regional Analysis</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Geographic sales performance and market analysis</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🗺️ Regional Performance")
        region_perf = analysis['regional_analysis']['region_performance']

        region_summary = region_perf.groupby('Region').agg(
            Total_Sales=('Total_Sales', 'sum'),
            Total_Profit=('Total_Profit', 'sum'),
            Order_Count=('Order_Count', 'sum'),
            Customer_Count=('Customer_Count', 'sum'),
            Avg_Margin=('Avg_Profit_Margin', 'mean')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=region_summary['Region'],
            y=region_summary['Total_Sales'],
            name='Sales',
            marker_color='#2e86ab',
            text=region_summary['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside',
            textfont=dict(color='#8892b0')
        ))
        fig.add_trace(go.Scatter(
            x=region_summary['Region'],
            y=region_summary['Total_Profit'],
            name='Profit',
            mode='lines+markers',
            line=dict(color='#64ffda', width=3),
            marker=dict(size=10),
            yaxis='y2'
        ))
        fig.update_layout(
            height=400,
            hovermode='x unified',
            yaxis2=dict(overlaying='y', side='right', gridcolor='#2a2d3a'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Region Profitability")
        region_profit = analysis['regional_analysis']['region_profitability']

        fig = go.Figure(go.Scatter(
            x=region_profit['Total_Sales'],
            y=region_profit['Avg_Margin'],
            mode='markers+text',
            marker=dict(
                size=region_profit['Total_Orders'] / 10,
                color=region_profit['Total_Profit'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title='Profit ($)'),
                line=dict(width=2, color='#1e2130')
            ),
            text=region_profit['Region'],
            textposition='top center',
            textfont=dict(color='#e0e0e0', size=11),
            hovertemplate='<b>%{text}</b><br>Sales: $%{x:,.2f}<br>Margin: %{y:.1f}%<br>Profit: $%{marker.color:,.2f}<extra></extra>'
        ))
        fig.update_layout(
            height=400,
            xaxis_title='Total Sales ($)',
            yaxis_title='Avg Profit Margin (%)'
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Regional trends
    st.markdown("### 📈 Regional Sales Trends")
    region_monthly = analysis['regional_analysis']['region_monthly']

    fig = go.Figure()
    for region in region_monthly['Region'].unique():
        region_data = region_monthly[region_monthly['Region'] == region]
        fig.add_trace(go.Scatter(
            x=region_data['Year_Month'],
            y=region_data['Total_Sales'],
            mode='lines',
            name=region,
            line=dict(width=2)
        ))
    fig.update_layout(
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Top states
    st.markdown("### 🏙️ Top States")
    top_states = analysis['regional_analysis']['top_states']

    fig = go.Figure(go.Bar(
        x=top_states['Total_Sales'],
        y=top_states['State'],
        orientation='h',
        marker=dict(
            color=top_states['Total_Sales'],
            colorscale='Blues',
            showscale=True
        ),
        text=top_states['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
        textposition='outside',
        textfont=dict(color='#8892b0')
    ))
    fig.update_layout(height=400, yaxis=dict(autorange='reversed'))
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def render_product_analysis():
    """Render product analysis page."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis

    st.markdown("<h1>📦 Product Analysis</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Product performance, categories, and profitability</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Category Overview", "🏆 Top Products", "📈 Profit Analysis"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            category_stats = analysis['category_analysis']['category_stats']
            fig = go.Figure(go.Bar(
                x=category_stats['Category'],
                y=category_stats['Total_Sales'],
                marker_color=[COLOR_MAP.get(c, '#2e86ab') for c in category_stats['Category']],
                text=category_stats['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                textfont=dict(color='#8892b0')
            ))
            fig.update_layout(height=350, title='Sales by Category')
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            subcat_stats = analysis['category_analysis']['subcategory_stats']
            fig = go.Figure(go.Bar(
                x=subcat_stats['Sub_Category'],
                y=subcat_stats['Total_Sales'],
                marker_color=[COLOR_MAP.get(c, '#2e86ab') for c in subcat_stats['Category']],
                text=subcat_stats['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                textfont=dict(color='#8892b0', size=9)
            ))
            fig.update_layout(height=350, xaxis_tickangle=45, title='Sales by Sub-Category')
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        # Category trends over time
        st.markdown("### 📈 Category Performance Over Time")
        cat_trends = analysis['category_analysis']['category_trends']

        fig = go.Figure()
        for cat in cat_trends['Category'].unique():
            cat_data = cat_trends[cat_trends['Category'] == cat]
            fig.add_trace(go.Scatter(
                x=cat_data['Year_Month'],
                y=cat_data['Total_Sales'],
                mode='lines',
                name=cat,
                line=dict(width=2.5)
            ))
        fig.update_layout(
            height=400,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 15 Products by Revenue")
            top_prods = analysis['product_analysis']['top_products'].head(15)

            fig = go.Figure(go.Bar(
                x=top_prods['Total_Sales'],
                y=top_prods['Product_Name'].str[:30] + '...',
                orientation='h',
                marker=dict(
                    color=top_prods['Total_Sales'],
                    colorscale='Blues',
                    showscale=True
                ),
                text=top_prods['Total_Sales'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                textfont=dict(color='#8892b0', size=9)
            ))
            fig.update_layout(height=500, yaxis=dict(autorange='reversed'))
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Product Performance Distribution")
            perf_dist = analysis['product_analysis']['performance_distribution']

            perf_order = ['Top', 'High', 'Medium', 'Low']
            perf_dist['Order'] = perf_dist['Performance'].map(
                {s: i for i, s in enumerate(perf_order)}
            )
            perf_dist = perf_dist.dropna().sort_values('Order')

            fig = go.Figure(go.Pie(
                labels=perf_dist['Performance'],
                values=perf_dist['Count'],
                hole=0.4,
                marker=dict(colors=['#ffd700', '#2e86ab', '#59a14f', '#e15759']),
                textinfo='label+percent',
                textfont=dict(color='#e0e0e0')
            ))
            fig.update_layout(height=400)
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            profit_by_cat = analysis['profit_analysis']['profit_by_category']
            fig = go.Figure(go.Bar(
                x=profit_by_cat['Category'],
                y=profit_by_cat['Total_Profit'],
                marker_color=[COLOR_MAP.get(c, '#2e86ab') for c in profit_by_cat['Category']],
                text=profit_by_cat['Total_Profit'].apply(lambda x: f'${x:,.0f}'),
                textposition='outside',
                textfont=dict(color='#8892b0')
            ))
            fig.update_layout(height=350, title='Profit by Category')
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            profit_discount = analysis['profit_analysis']['profit_discount_analysis']
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=profit_discount['Discount_Range'],
                y=profit_discount['Total_Sales'],
                name='Sales',
                marker_color='#2e86ab'
            ))
            fig.add_trace(go.Scatter(
                x=profit_discount['Discount_Range'],
                y=profit_discount['Avg_Margin'],
                name='Margin %',
                mode='lines+markers',
                line=dict(color='#64ffda', width=3),
                yaxis='y2'
            ))
            fig.update_layout(
                height=350,
                title='Sales & Margin by Discount Range',
                xaxis_tickangle=45,
                yaxis2=dict(overlaying='y', side='right', gridcolor='#2a2d3a')
            )
            fig = apply_plotly_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


def render_forecasting():
    """Render ML forecasting page."""
    df = st.session_state.filtered_df

    st.markdown("<h1>🤖 ML Forecasting</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">AI-powered sales prediction and trend forecasting</p>',
                unsafe_allow_html=True)

    st.info("""
    **Forecasting Models Available:**
    - **Prophet**: Facebook's time series forecasting (handles seasonality & holidays well)
    - **XGBoost**: Gradient boosting with engineered features (captures complex patterns)
    """)

    if st.button("🚀 Run Forecasting Models", type="primary", use_container_width=True):
        with st.spinner("Running ML forecasting models... This may take a moment."):
            forecasting_results = run_forecasting(df)
            st.session_state.forecasting_results = forecasting_results
        st.success("Forecasting complete!")

    if 'forecasting_results' in st.session_state:
        results = st.session_state.forecasting_results

        tab1, tab2, tab3 = st.tabs(["📈 Prophet Forecast", "📊 XGBoost Forecast", "⚠️ Anomaly Detection"])

        with tab1:
            prophet = results.get('prophet', {})
            future = prophet.get('future_forecast')

            if future is not None and not future.empty:
                st.markdown("### 📈 Prophet Sales Forecast")

                col1, col2, col3 = st.columns(3)
                with col1:
                    accuracy = prophet.get('accuracy', {})
                    mape = accuracy.get('mape', 0)
                    st.metric("MAPE", f"{mape:.1f}%", "Lower is better")
                with col2:
                    rmse = accuracy.get('rmse', 0)
                    st.metric("RMSE", f"${rmse:.2f}")
                with col3:
                    st.metric("Forecast Periods", f"{len(future)} months")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=future['Order_Date'],
                    y=future['Sales_Forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#2e86ab', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.1)'
                ))
                if 'Lower_Bound' in future.columns and 'Upper_Bound' in future.columns:
                    fig.add_trace(go.Scatter(
                        x=future['Order_Date'],
                        y=future['Upper_Bound'],
                        mode='lines',
                        name='Upper Bound',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=future['Order_Date'],
                        y=future['Lower_Bound'],
                        mode='lines',
                        name='Confidence Interval',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(46, 134, 171, 0.2)'
                    ))
                fig.update_layout(
                    height=400,
                    hovermode='x unified',
                    title='Sales Forecast (Next 6 Months)'
                )
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Trend component
                trend = prophet.get('trend')
                if trend is not None:
                    st.markdown("### 📉 Trend Component")
                    fig = go.Figure(go.Scatter(
                        x=trend['Order_Date'],
                        y=trend['Trend'],
                        mode='lines',
                        line=dict(color='#64ffda', width=3)
                    ))
                    fig.update_layout(height=300, hovermode='x unified')
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)

        with tab2:
            xgb_result = results.get('xgboost', {})
            future_xgb = xgb_result.get('future_forecast')

            if future_xgb is not None and not future_xgb.empty:
                st.markdown("### 📊 XGBoost Sales Forecast")

                accuracy = xgb_result.get('accuracy', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"${accuracy.get('mae', 0):.2f}")
                with col2:
                    st.metric("RMSE", f"${accuracy.get('rmse', 0):.2f}")
                with col3:
                    st.metric("R²", f"{accuracy.get('r2', 0):.3f}")
                with col4:
                    st.metric("MAPE", f"{accuracy.get('mape', 0):.1f}%")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=future_xgb['Order_Date'],
                    y=future_xgb['Sales_Forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#f28e2b', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(242, 142, 43, 0.1)'
                ))
                if 'Lower_Bound' in future_xgb.columns and 'Upper_Bound' in future_xgb.columns:
                    fig.add_trace(go.Scatter(
                        x=future_xgb['Order_Date'],
                        y=future_xgb['Upper_Bound'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_xgb['Order_Date'],
                        y=future_xgb['Lower_Bound'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(242, 142, 43, 0.2)'
                    ))
                fig.update_layout(height=400, hovermode='x unified')
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Feature importance
                feature_imp = xgb_result.get('feature_importance')
                if feature_imp is not None:
                    st.markdown("### 🔍 Feature Importance")
                    fig = go.Figure(go.Bar(
                        x=feature_imp['Importance'].head(10),
                        y=feature_imp['Feature'].head(10),
                        orientation='h',
                        marker=dict(color=feature_imp['Importance'].head(10), colorscale='Viridis'),
                        text=feature_imp['Importance'].head(10).apply(lambda x: f'{x:.3f}'),
                        textposition='outside',
                        textfont=dict(color='#8892b0')
                    ))
                    fig.update_layout(height=400, yaxis=dict(autorange='reversed'))
                    fig = apply_plotly_theme(fig)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("XGBoost forecast not available. The model may need more data or different parameters.")

        with tab3:
            anomalies = results.get('anomalies')
            if anomalies is not None and not anomalies.empty:
                st.markdown(f"### ⚠️ Detected Anomalies: {len(anomalies)}")

                # Timeline with anomalies highlighted
                daily_sales = df.groupby('Order_Date')['Sales'].sum().reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_sales['Order_Date'],
                    y=daily_sales['Sales'],
                    mode='lines',
                    name='Daily Sales',
                    line=dict(color='#2e86ab', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=anomalies['Order_Date'] if 'Order_Date' in anomalies.columns else [],
                    y=anomalies['Sales'] if 'Sales' in anomalies.columns else [],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='#e15759',
                        size=12,
                        symbol='x',
                        line=dict(width=2, color='white')
                    )
                ))
                fig.update_layout(height=400, hovermode='x unified')
                fig = apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

                # Display anomalies table
                st.markdown("#### Anomaly Details")
                display_cols = [c for c in ['Order_Date', 'Sales', 'Orders', 'Anomaly_Type']
                               if c in anomalies.columns]
                st.dataframe(
                    anomalies[display_cols].sort_values('Sales', ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No significant anomalies detected in the dataset.")

        # Business insights
        insights = results.get('insights', [])
        if insights:
            st.markdown("### 💡 AI-Generated Business Insights")
            cols = st.columns(2)
            for i, insight in enumerate(insights):
                col_idx = i % 2
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="insight-card">
                        <p style="margin: 0; color: #e0e0e0; font-size: 0.9rem;">{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Click 'Run Forecasting Models' to generate predictions and insights.")


def render_business_insights():
    """Render business insights page."""
    df = st.session_state.filtered_df
    analysis = st.session_state.analysis

    st.markdown("<h1>💡 Business Insights</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">AI-powered analytics and actionable recommendations</p>',
                unsafe_allow_html=True)

    # Generate insights
    kpis = analysis['kpis']

    # Key metrics dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Profitability Index", 
            f"{kpis['Profit_Margin']:.1f}%",
            "Target: >20%" if kpis['Profit_Margin'] >= 20 else "Needs Improvement"
        )
    with col2:
        cust_ratio = kpis['Revenue_per_Customer'] / kpis['Average_Order_Value'] if kpis['Average_Order_Value'] > 0 else 0
        st.metric(
            "Customer Lifetime Value",
            format_currency(kpis['Revenue_per_Customer']),
            f"{cust_ratio:.1f}x AOV"
        )
    with col3:
        st.metric(
            "Order-to-Customer Ratio",
            f"{kpis['Orders_per_Customer']:.2f}",
            "Repeat Purchase Rate"
        )
    with col4:
        st.metric(
            "Average Discount",
            f"{kpis['Average_Discount']*100:.1f}%",
            f"Impact: {format_currency(kpis['Total_Discount_Impact'])}"
        )

    st.divider()

    # AI Insights
    st.markdown("### 🤖 AI-Generated Insights")

    forecaster = SalesForecaster(df)
    insights = forecaster.generate_business_insights()

    for i, insight in enumerate(insights):
        if insight.startswith("💡"):
            st.markdown(f"#### {insight}")
        else:
            st.markdown(f"""
            <div class="insight-card">
                <p style="margin: 0; color: #e0e0e0; font-size: 0.95rem;">{insight}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Strategic Recommendations
    st.markdown("### 🎯 Strategic Recommendations")

    recommendations = [
        {
            "title": "Revenue Growth",
            "strategy": "Focus on top-performing categories and customer segments",
            "action": "Increase marketing budget for Technology products targeting Platinum/Gold segments",
            "impact": "Potential 20-30% revenue increase"
        },
        {
            "title": "Profit Optimization",
            "strategy": "Optimize discount strategy to protect margins",
            "action": "Reduce discounts on high-demand products, bundle slow movers",
            "impact": "Potential 5-10% margin improvement"
        },
        {
            "title": "Customer Retention",
            "strategy": "Implement targeted loyalty programs",
            "action": "Create VIP program for top 20% customers with exclusive benefits",
            "impact": "Expected 15-25% increase in repeat purchases"
        },
        {
            "title": "Market Expansion",
            "strategy": "Expand successful regional strategies",
            "action": "Replicate top region's marketing approach in underperforming areas",
            "impact": "Balanced regional growth, 10-15% total sales increase"
        },
        {
            "title": "Inventory Optimization",
            "strategy": "Use seasonal patterns for inventory planning",
            "action": "Stock up 2 months before peak seasons, offer pre-season discounts",
            "impact": "Reduced carrying costs, improved cash flow"
        }
    ]

    cols = st.columns(3)
    for i, rec in enumerate(recommendations):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e2130, #252840);
                        border: 1px solid #2e3145;
                        border-radius: 16px;
                        padding: 20px;
                        height: 220px;
                        margin-bottom: 16px;">
                <h4 style="color: #64ffda; margin: 0 0 8px 0;">{rec['title']}</h4>
                <p style="color: #8892b0; font-size: 0.85rem; margin: 4px 0;">
                    <strong>Strategy:</strong> {rec['strategy']}</p>
                <p style="color: #e0e0e0; font-size: 0.85rem; margin: 4px 0;">
                    <strong>Action:</strong> {rec['action']}</p>
                <p style="color: #48bfe3; font-size: 0.85rem; margin: 4px 0;">
                    <strong>Impact:</strong> {rec['impact']}</p>
            </div>
            """, unsafe_allow_html=True)


def render_reports():
    """Render reports page with export options."""
    df = st.session_state.df
    analysis = st.session_state.analysis

    st.markdown("<h1>📋 Reports & Export</h1>", unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Generate and download comprehensive business reports</p>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Generate Reports")
        st.markdown("""
        Generate comprehensive business reports including:
        - Executive summary with KPIs
        - Sales trends and analysis
        - Customer insights
        - Product performance
        - Profitability analysis

        Reports are saved to the `reports/` directory.
        """)

        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                report_gen = ReportGenerator(df, analysis)
                path = report_gen.save_pdf_report()
                if path:
                    st.success(f"✅ PDF report saved to: `{path}`")
                    with open(path, 'rb') as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=path.name,
                            mime="application/pdf",
                            use_container_width=True
                        )
                else:
                    st.error("PDF generation failed. Check the logs for details.")

    with col2:
        st.markdown("### 📈 Export Data")
        st.markdown("""
        Export data in various formats:
        - **CSV Summary**: Condensed business report
        - **Full Data Export**: Complete dataset with all features
        - **Business Insights**: AI-generated insights and recommendations
        """)

        if st.button("📥 Export CSV Summary", use_container_width=True):
            report_gen = ReportGenerator(df, analysis)
            csv_content = report_gen.generate_csv_summary()
            st.download_button(
                label="📥 Download CSV Summary",
                data=csv_content,
                file_name=f"sales_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ CSV summary generated!")

        if st.button("📥 Export Full Data", use_container_width=True):
            report_gen = ReportGenerator(df, analysis)
            csv_content = report_gen.generate_csv_export()
            st.download_button(
                label="📥 Download Full Data (CSV)",
                data=csv_content,
                file_name=f"sales_data_full_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success("✅ Full data export generated!")

        if st.button("💡 Export Business Insights", use_container_width=True):
            report_gen = ReportGenerator(df, analysis)
            content = report_gen.generate_insights_summary()
            st.download_button(
                label="📥 Download Insights (TXT)",
                data=content,
                file_name=f"business_insights_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.success("✅ Insights report generated!")

    st.divider()

    # Preview data
    st.markdown("### 👁️ Data Preview")
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    # Dataset stats
    st.markdown("### 📊 Dataset Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point."""

    # Initialize session state
    if 'initialized' not in st.session_state:
        with st.spinner("🚀 Initializing AI Sales Analytics Dashboard..."):
            # Load data
            df = load_sales_data()
            st.session_state.df = df
            st.session_state.filtered_df = df

            # Run analysis
            analysis = run_analysis(df)
            st.session_state.analysis = analysis

            # Initialize SQL engine
            try:
                engine = init_sql_engine(df)
                st.session_state.engine = engine
            except Exception as e:
                logger.error(f"SQL engine init failed: {e}")
                st.session_state.engine = None

            st.session_state.initialized = True

        st.success("✅ Dashboard initialized successfully!")

    # Render sidebar and get navigation
    selected = render_sidebar()

    # Render selected page
    if selected == "📈 Overview":
        render_overview()
    elif selected == "📊 Sales Analytics":
        render_sales_analytics()
    elif selected == "👥 Customer Insights":
        render_customer_insights()
    elif selected == "🌍 Regional Analysis":
        render_regional_analysis()
    elif selected == "📦 Product Analysis":
        render_product_analysis()
    elif selected == "🤖 ML Forecasting":
        render_forecasting()
    elif selected == "💡 Business Insights":
        render_business_insights()
    elif selected == "📋 Reports":
        render_reports()

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🤖 AI-Powered Sales Analytics Dashboard | Built with Streamlit, Python & Machine Learning</p>
        <p>Data Analyst Portfolio Project © 2024</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
