#!/usr/bin/env python3
"""
FUNDS DASHBOARD WITH PORTFOLIO CART
Dashboard original con carrito de portafolios integrado
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os

import cvxpy as cp
import base64
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')

# PORTFOLIO CART INTEGRATION
from portfolio_cart import PortfolioCart

# Page configuration
st.set_page_config(
    page_title="Dashboard de Análisis de Fondos + Carrito",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional dark theme styling
st.markdown("""
<style>
    /* Dark theme optimized styles */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 1rem;
        text-align: center;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #404040;
    }
    .performance-table {
        font-size: 0.9rem;
    }
    .positive-return {
        color: #10b981;
        font-weight: 500;
    }
    .negative-return {
        color: #ef4444;
        font-weight: 500;
    }
    .fund-row {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .cart-highlight {
        border: 2px solid #3b82f6 !important;
        background-color: #1e40af !important;
    }
    
    /* Better contrast for charts */
    .js-plotly-plot {
        background-color: transparent !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0f172a;
    }
    
    /* Custom button styling */
    .stButton > button {
        border-radius: 0.5rem;
        border: 1px solid #475569;
        background-color: #334155;
        color: #f1f5f9;
    }
    
    .stButton > button:hover {
        border-color: #3b82f6;
        background-color: #1e40af;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load funds data and dictionary from CSV files"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        funds_path = os.path.join(data_dir, 'funds_prices.csv')
        funds = pd.read_csv(funds_path)
        
        dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
        etf_dict = pd.read_csv(dict_path)
        
        return funds, etf_dict
        
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.info("Please run 'python convert_data.py' first to generate the required CSV files.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def calculate_performance_metrics(funds_df, fund_ticker):
    """Calculate comprehensive performance metrics for a fund"""
    try:
        if fund_ticker not in funds_df.columns:
            return None
        
        prices = funds_df[['Dates', fund_ticker]].dropna()
        if len(prices) < 2:
            return None
        
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        prices['Returns'] = prices[fund_ticker].pct_change()
        
        current_date = prices['Dates'].max()
        current_year = current_date.year
        
        # YTD Return
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        ytd_return = ((ytd_data[fund_ticker].iloc[-1] / ytd_data[fund_ticker].iloc[0]) - 1) * 100 if len(ytd_data) > 1 else 0
        
        # Monthly Return (last 30 days)
        month_start = current_date - timedelta(days=30)
        month_data = prices[prices['Dates'] >= month_start]
        monthly_return = ((month_data[fund_ticker].iloc[-1] / month_data[fund_ticker].iloc[0]) - 1) * 100 if len(month_data) > 1 else 0
        
        # 1 Year Return
        year_1_start = current_date - timedelta(days=365)
        year_1_data = prices[prices['Dates'] >= year_1_start]
        return_1y = ((year_1_data[fund_ticker].iloc[-1] / year_1_data[fund_ticker].iloc[0]) - 1) * 100 if len(year_1_data) > 1 else 0
        
        # Annual returns for specific years
        returns_by_year = {}
        for year in [2024, 2023, 2022]:
            year_start = pd.to_datetime(f'{year}-01-01')
            year_end = pd.to_datetime(f'{year}-12-31')
            year_data = prices[(prices['Dates'] >= year_start) & (prices['Dates'] <= year_end)]
            if len(year_data) > 1:
                year_return = ((year_data[fund_ticker].iloc[-1] / year_data[fund_ticker].iloc[0]) - 1) * 100
                returns_by_year[f'{year} Return (%)'] = year_return
        
        # Volatility (annualized)
        volatility = prices['Returns'].std() * np.sqrt(252) * 100
        
        # Max Drawdown
        cumulative = (1 + prices['Returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # VaR and CVaR (5% confidence level, annualized)
        returns_clean = prices['Returns'].dropna()
        if len(returns_clean) > 0:
            var_5 = np.percentile(returns_clean, 5) * np.sqrt(252) * 100
            cvar_5 = returns_clean[returns_clean <= np.percentile(returns_clean, 5)].mean() * np.sqrt(252) * 100
        else:
            var_5 = 0
            cvar_5 = 0
        
        metrics = {
            'YTD Return (%)': ytd_return,
            'Monthly Return (%)': monthly_return,
            '1Y Return (%)': return_1y,
            'Volatility (%)': volatility,
            'Max Drawdown (%)': max_drawdown,
            'VaR 5% (%)': var_5,
            'CVaR 5% (%)': cvar_5
        }
        
        metrics.update(returns_by_year)
        
        return metrics
        
    except Exception as e:
        return None

def calculate_custom_score(df, weights):
    """Calculate custom score based on user-defined weights"""
    df_scored = df.copy()
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        return df_scored
    
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Calculate score for each fund
    scores = []
    for _, row in df.iterrows():
        score = 0
        for metric, weight in normalized_weights.items():
            if metric in row and pd.notna(row[metric]):
                value = row[metric]
                
                # For negative metrics (drawdown, volatility, VaR, CVaR), invert the score
                if 'Drawdown' in metric or 'Volatility' in metric or 'VaR' in metric or 'CVaR' in metric:
                    # Convert to positive score (less negative is better)
                    normalized_value = max(0, 100 + value)  # Since these are negative
                else:
                    # For positive metrics (returns), use as is
                    normalized_value = max(0, value + 100)  # Add 100 to handle negative returns
                
                score += normalized_value * weight
        
        scores.append(score)
    
    df_scored['Custom Score'] = scores
    df_scored = df_scored.sort_values('Custom Score', ascending=False).reset_index(drop=True)
    
    return df_scored

def create_cumulative_returns_chart(funds_df, selected_funds, start_date, end_date):
    """Create cumulative returns chart for selected funds"""
    try:
        funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
        filtered_data = funds_df[(funds_df['Dates'] >= start_date) & (funds_df['Dates'] <= end_date)].copy()
        
        if filtered_data.empty:
            return None
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, fund in enumerate(selected_funds):
            if fund in filtered_data.columns:
                fund_data = filtered_data[['Dates', fund]].dropna()
                if len(fund_data) > 1:
                    # Calculate cumulative returns (normalized to 100 at start)
                    base_price = fund_data[fund].iloc[0]
                    cumulative_returns = (fund_data[fund] / base_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=fund_data['Dates'],
                        y=cumulative_returns,
                        mode='lines',
                        name=fund,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{fund}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title=f"Cumulative Returns Comparison ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Base 100)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#404040', color='#fafafa'),
            yaxis=dict(gridcolor='#404040', color='#fafafa'),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#404040', borderwidth=1)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def main():
    """Main dashboard function with integrated portfolio cart"""
    
    # INTEGRAR CARRITO DE PORTAFOLIOS
    PortfolioCart.initialize()
    
    # Verificar si mostrar la pestaña de gestión de portafolio
    if st.session_state.get('show_portfolio_tab', False):
        PortfolioCart.render_portfolio_management_tab()
        return
    
    # Load data
    with st.spinner("Loading data..."):
        funds_data, etf_dict = load_data()
    
    if funds_data is None or etf_dict is None:
        return
    
    # RENDERIZAR WIDGET DEL CARRITO EN SIDEBAR
    PortfolioCart.render_cart_widget()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Dashboard de Análisis de Fondos + Carrito de Portafolios</h1>', unsafe_allow_html=True)
    
    # SIDEBAR FILTERS
    st.sidebar.markdown("## 🔍 Filtros de Análisis")
    
    # Prepare data for filters
    funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Create fund options dictionary for better display
    fund_options = {}
    for ticker in fund_columns:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        if not fund_info.empty and 'Fund Name' in fund_info.columns:
            fund_name = fund_info['Fund Name'].iloc[0]
            fund_options[f"{fund_name} ({ticker})"] = ticker
        else:
            fund_options[ticker] = ticker
    
    # Category filter
    if 'Category' in etf_dict.columns:
        categories = ['All'] + sorted(etf_dict['Category'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Category:", categories)
        
        if selected_category != 'All':
            filtered_tickers = etf_dict[etf_dict['Category'] == selected_category]['Ticker'].tolist()
            fund_options = {k: v for k, v in fund_options.items() if v in filtered_tickers}
    
    # Search filter
    search_term = st.sidebar.text_input("🔍 Search fund:", "")
    if search_term:
        fund_options = {k: v for k, v in fund_options.items() if search_term.lower() in k.lower()}
    
    # Performance filter
    st.sidebar.markdown("### 📊 Performance Filters")
    min_ytd = st.sidebar.number_input("Min YTD Return (%):", value=-100.0, step=1.0)
    min_1y = st.sidebar.number_input("Min 1Y Return (%):", value=-100.0, step=1.0)
    max_volatility = st.sidebar.number_input("Max Volatility (%):", value=100.0, step=1.0)
    
    # Scoring weights configuration
    st.sidebar.markdown("## ⚖️ Scoring Weights")
    
    weights = {}
    weights['YTD Return (%)'] = st.sidebar.slider("YTD Return", 0, 100, 20)
    weights['1Y Return (%)'] = st.sidebar.slider("1Y Return", 0, 100, 25)
    weights['2024 Return (%)'] = st.sidebar.slider("2024 Return", 0, 100, 15)
    weights['Max Drawdown (%)'] = st.sidebar.slider("Max Drawdown", 0, 100, 15)
    weights['Volatility (%)'] = st.sidebar.slider("Volatility", 0, 100, 10)
    weights['VaR 5% (%)'] = st.sidebar.slider("VaR 5%", 0, 100, 10)
    weights['CVaR 5% (%)'] = st.sidebar.slider("CVaR 5%", 0, 100, 5)
    
    # Chart date configuration
    min_date = funds_data['Dates'].min().date()
    max_date = funds_data['Dates'].max().date()
    
    # Quick date shortcuts
    st.sidebar.markdown("**Quick Shortcuts:**")
    shortcut_cols = st.sidebar.columns(2)
    
    with shortcut_cols[0]:
        if st.button("YTD", use_container_width=True):
            st.session_state.chart_start_date = pd.to_datetime(f'{max_date.year}-01-01').date()
            st.session_state.chart_end_date = max_date
            st.rerun()
        if st.button("1 Year", use_container_width=True):
            st.session_state.chart_start_date = max_date - pd.Timedelta(days=365)
            st.session_state.chart_end_date = max_date
            st.rerun()
    
    with shortcut_cols[1]:
        if st.button("2 Years", use_container_width=True):
            st.session_state.chart_start_date = max_date - pd.Timedelta(days=730)
            st.session_state.chart_end_date = max_date
            st.rerun()
        if st.button("1 Month", use_container_width=True):
            st.session_state.chart_start_date = max_date - pd.Timedelta(days=30)
            st.session_state.chart_end_date = max_date
            st.rerun()
    
    # Initialize session state for dates if not exists
    if 'chart_start_date' not in st.session_state:
        st.session_state.chart_start_date = max_date - pd.Timedelta(days=365*2)
    if 'chart_end_date' not in st.session_state:
        st.session_state.chart_end_date = max_date
    
    # Date inputs
    chart_start_date = st.sidebar.date_input(
        "Chart Start Date:",
        value=st.session_state.chart_start_date,
        min_value=min_date,
        max_value=max_date,
        key="start_date_input"
    )
    
    chart_end_date = st.sidebar.date_input(
        "Chart End Date:",
        value=st.session_state.chart_end_date,
        min_value=min_date,
        max_value=max_date,
        key="end_date_input"
    )
    
    # Update session state and trigger rerun if dates changed
    date_changed = False
    if st.session_state.chart_start_date != chart_start_date:
        st.session_state.chart_start_date = chart_start_date
        date_changed = True
    if st.session_state.chart_end_date != chart_end_date:
        st.session_state.chart_end_date = chart_end_date
        date_changed = True
    
    # Force rerun if dates changed to ensure chart updates
    if date_changed:
        st.rerun()
    
    # MAIN CONTENT
    filtered_funds = list(fund_options.values())
    
    if not filtered_funds:
        st.warning("No funds match the selected filter criteria.")
        return
    
    # Calculate performance metrics
    with st.spinner("Calculating performance metrics..."):
        performance_data = []
        
        for ticker in filtered_funds:
            metrics = calculate_performance_metrics(funds_data, ticker)
            if metrics:
                # Get fund name
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
                
                # Apply performance filters
                if (metrics['YTD Return (%)'] >= min_ytd and 
                    metrics['1Y Return (%)'] >= min_1y and 
                    metrics['Volatility (%)'] <= max_volatility):
                    
                    row = {'Ticker': ticker, 'Fund Name': fund_name}
                    row.update(metrics)
                    performance_data.append(row)
        
        if not performance_data:
            st.error("No funds meet the performance criteria.")
            return
        
        df_performance = pd.DataFrame(performance_data)
    
    # Calculate custom score
    df_scored = calculate_custom_score(df_performance, weights)
    
    # DISPLAY RESULTS WITH CART BUTTONS
    st.markdown("## 🏆 Fund Rankings with Portfolio Cart")
    st.markdown(f"Showing {len(df_scored)} funds ordered by custom score. **Click '🛒 Agregar' to add funds to your portfolio cart.**")
    
    # Prepare display dataframe
    display_df = df_scored.copy()
    
    # Format percentage columns
    percentage_cols = ['YTD Return (%)', 'Monthly Return (%)', '1Y Return (%)', 
                      '2024 Return (%)', '2023 Return (%)', '2022 Return (%)',
                      'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
    
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    if 'Custom Score' in display_df.columns:
        display_df['Custom Score'] = display_df['Custom Score'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
    
    # MOSTRAR FONDOS CON BOTONES DE CARRITO
    for idx, row in display_df.head(20).iterrows():  # Show top 20
        # Verificar si está en el carrito para destacar
        is_in_cart = row['Ticker'] in st.session_state.get('portfolio_cart', {})
        container_class = "cart-highlight" if is_in_cart else "fund-row"
        
        with st.container():
            st.markdown(f'<div class="{container_class}">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Fund information
                score_text = f"**Score: {row['Custom Score']}**" if 'Custom Score' in row else ""
                st.markdown(f"**{row['Fund Name']}** ({row['Ticker']}) {score_text}")
                
                # Key metrics
                metrics_text = []
                if 'YTD Return (%)' in row:
                    metrics_text.append(f"YTD: {row['YTD Return (%)']}")
                if '1Y Return (%)' in row:
                    metrics_text.append(f"1Y: {row['1Y Return (%)']}")
                if 'Volatility (%)' in row:
                    metrics_text.append(f"Vol: {row['Volatility (%)']}")
                if 'Max Drawdown (%)' in row:
                    metrics_text.append(f"DD: {row['Max Drawdown (%)']}")
                
                if metrics_text:
                    st.caption(" | ".join(metrics_text))
            
            with col2:
                # BOTÓN PARA AGREGAR AL CARRITO
                PortfolioCart.render_add_button(
                    row['Ticker'], 
                    row['Fund Name'], 
                    key_suffix=f"ranking_{idx}"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
    
    # Show complete table for reference
    with st.expander("📊 Complete Data Table"):
        final_cols = ['Ticker', 'Fund Name', 'Custom Score'] + percentage_cols
        final_cols = [col for col in final_cols if col in display_df.columns]
        st.dataframe(display_df[final_cols], use_container_width=True)
    
    # CUMULATIVE RETURNS CHART
    st.markdown("## 📊 Chart Analysis")
    
    # Fund selector for chart
    top_funds = list(df_scored.head(10)['Ticker'])
    fund_names_for_chart = {}
    for ticker in top_funds:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
        fund_names_for_chart[f"{fund_name} ({ticker})"] = ticker
    
    selected_fund_names = st.multiselect(
        "Select funds for chart analysis:",
        options=list(fund_names_for_chart.keys()),
        default=list(fund_names_for_chart.keys())[:5] if len(fund_names_for_chart) >= 5 else list(fund_names_for_chart.keys())
    )
    
    # Get selected fund tickers
    selected_funds = [fund_names_for_chart[name] for name in selected_fund_names]
    
    if selected_funds:
        st.markdown(f"**Showing {len(selected_funds)} selected funds from {st.session_state.chart_start_date} to {st.session_state.chart_end_date}**")
        
        # Create and display chart with custom date range
        chart_key = f"chart_{st.session_state.chart_start_date}_{st.session_state.chart_end_date}_{len(selected_funds)}"
        
        chart = create_cumulative_returns_chart(
            funds_data, 
            selected_funds, 
            pd.to_datetime(st.session_state.chart_start_date), 
            pd.to_datetime(st.session_state.chart_end_date)
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True, key=chart_key)
    else:
        st.info("👆 Select funds from the list above to view cumulative returns chart.")

if __name__ == "__main__":
    main()