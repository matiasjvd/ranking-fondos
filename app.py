#!/usr/bin/env python3
"""
FUND ANALYSIS DASHBOARD
Professional fund analysis dashboard with portfolio management capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cvxpy as cp
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fund Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class PortfolioManager:
    """Portfolio management functionality"""
    
    @staticmethod
    def initialize():
        """Initialize portfolio state"""
        if 'selected_funds' not in st.session_state:
            st.session_state.selected_funds = set()
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'show_portfolio_analysis' not in st.session_state:
            st.session_state.show_portfolio_analysis = False
    
    @staticmethod
    def render_fund_selector(ticker, fund_name, key_suffix=""):
        """Render fund selector for portfolio"""
        PortfolioManager.initialize()
        
        is_selected = ticker in st.session_state.selected_funds
        
        if st.button(
            f"{'✓ ' if is_selected else '+ '}{fund_name}",
            key=f"select_{ticker}_{key_suffix}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            if ticker in st.session_state.selected_funds:
                # Remove from portfolio
                st.session_state.selected_funds.discard(ticker)
                if ticker in st.session_state.portfolio_weights:
                    del st.session_state.portfolio_weights[ticker]
                
                # Rebalance remaining funds
                remaining_funds = len(st.session_state.selected_funds)
                if remaining_funds > 0:
                    equal_weight = 100.0 / remaining_funds
                    for remaining_ticker in st.session_state.selected_funds:
                        st.session_state.portfolio_weights[remaining_ticker] = equal_weight
            else:
                # Add to portfolio
                st.session_state.selected_funds.add(ticker)
                if ticker not in st.session_state.portfolio_weights:
                    num_funds = len(st.session_state.selected_funds)
                    equal_weight = 100.0 / num_funds if num_funds > 0 else 0
                    st.session_state.portfolio_weights[ticker] = equal_weight
                    
                    # Rebalance other funds
                    for other_ticker in st.session_state.selected_funds:
                        st.session_state.portfolio_weights[other_ticker] = equal_weight
            
            st.rerun()
    
    @staticmethod
    def render_portfolio_sidebar():
        """Render portfolio widget in sidebar"""
        PortfolioManager.initialize()
        
        st.sidebar.markdown("## Portfolio Manager")
        
        num_funds = len(st.session_state.selected_funds)
        
        if num_funds == 0:
            st.sidebar.info("No assets selected")
            return
        
        st.sidebar.metric("Selected Assets", num_funds)
        
        # List funds with weights
        st.sidebar.markdown("### Asset Allocation:")
        for ticker in st.session_state.selected_funds:
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
        
        # Check total weights
        total_weight = sum(st.session_state.portfolio_weights.values())
        if abs(total_weight - 100) > 0.1:
            st.sidebar.warning(f"Total: {total_weight:.1f}%")
        else:
            st.sidebar.success("Total: 100%")
        
        # Action buttons
        if st.sidebar.button("Analyze Portfolio", use_container_width=True):
            st.session_state.show_portfolio_analysis = True
            st.rerun()
        
        if st.sidebar.button("Clear Portfolio", use_container_width=True):
            st.session_state.selected_funds.clear()
            st.session_state.portfolio_weights.clear()
            st.rerun()

@st.cache_data
def load_data():
    """Load funds data and dictionary"""
    try:
        funds_data = pd.read_csv('data/funds_prices.csv')
        etf_dict = pd.read_csv('data/funds_dictionary.csv')
        return funds_data, etf_dict
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def calculate_custom_score(df, weights):
    """Calculate custom score for funds"""
    df_scored = df.copy()
    
    # Normalize metrics (higher is better for returns, lower is better for risk)
    score_components = {}
    
    # Returns (higher is better)
    return_cols = ['YTD Return (%)', '1Y Return (%)', '2024 Return (%)']
    for col in return_cols:
        if col in df.columns:
            normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            score_components[col] = normalized.fillna(0)
    
    # Risk metrics (lower is better, so we invert)
    risk_cols = ['Volatility (%)', 'Max Drawdown (%)']
    for col in risk_cols:
        if col in df.columns:
            # Invert so lower risk = higher score
            normalized = 1 - ((df[col] - df[col].min()) / (df[col].max() - df[col].min()))
            score_components[col] = normalized.fillna(0)
    
    # Calculate weighted score
    scores = pd.Series(0, index=df.index)
    total_weight = 0
    
    for col, weight in weights.items():
        if col in score_components:
            scores += score_components[col] * weight
            total_weight += weight
    
    if total_weight > 0:
        scores = (scores / total_weight) * 100
    
    df_scored['Custom Score'] = scores
    df_scored = df_scored.sort_values('Custom Score', ascending=False).reset_index(drop=True)
    
    return df_scored

def create_cumulative_returns_chart(funds_df, selected_funds, start_date, end_date, fund_names_dict=None):
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
                    
                    # Get display name
                    display_name = fund
                    if fund_names_dict:
                        for key, ticker in fund_names_dict.items():
                            if ticker == fund:
                                display_name = key.split(' (')[0]
                                break
                    
                    fig.add_trace(go.Scatter(
                        x=fund_data['Dates'],
                        y=cumulative_returns,
                        mode='lines',
                        name=display_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{display_name}</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                    ))
        
        fig.update_layout(
            title=f"Cumulative Returns Comparison ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Base 100)",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def calculate_fund_metrics(funds_data, ticker):
    """Calculate individual fund metrics"""
    try:
        if ticker not in funds_data.columns:
            return None
        
        prices = funds_data[['Dates', ticker]].dropna()
        if len(prices) < 2:
            return None
        
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        prices['Returns'] = prices[ticker].pct_change()
        
        current_date = prices['Dates'].max()
        current_year = current_date.year
        
        # YTD Return
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        ytd_return = ((ytd_data[ticker].iloc[-1] / ytd_data[ticker].iloc[0]) - 1) * 100 if len(ytd_data) > 1 else 0
        
        # Monthly Return (last 30 days)
        month_start = current_date - timedelta(days=30)
        month_data = prices[prices['Dates'] >= month_start]
        monthly_return = ((month_data[ticker].iloc[-1] / month_data[ticker].iloc[0]) - 1) * 100 if len(month_data) > 1 else 0
        
        # 1 Year Return
        year_1_start = current_date - timedelta(days=365)
        year_1_data = prices[prices['Dates'] >= year_1_start]
        return_1y = ((year_1_data[ticker].iloc[-1] / year_1_data[ticker].iloc[0]) - 1) * 100 if len(year_1_data) > 1 else 0
        
        # Annual returns for specific years
        returns_by_year = {}
        for year in [2024, 2023, 2022]:
            year_start = pd.to_datetime(f'{year}-01-01')
            year_end = pd.to_datetime(f'{year}-12-31')
            year_data = prices[(prices['Dates'] >= year_start) & (prices['Dates'] <= year_end)]
            if len(year_data) > 1:
                year_return = ((year_data[ticker].iloc[-1] / year_data[ticker].iloc[0]) - 1) * 100
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

def main():
    """Main dashboard function"""
    
    # Integrate portfolio manager
    PortfolioManager.render_portfolio_sidebar()
    
    # Check if portfolio analysis should be shown
    if st.session_state.get('show_portfolio_analysis', False):
        render_portfolio_analysis()
        return
    
    # Load data
    with st.spinner("Loading data..."):
        funds_data, etf_dict = load_data()
    
    if funds_data is None or etf_dict is None:
        return
    
    # Header
    st.markdown('<h1 class="main-header">Fund Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar filters
    st.sidebar.markdown("## Analysis Filters")
    
    # Prepare data for filters
    funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Date range filter
    min_date = funds_data['Dates'].min().date()
    max_date = funds_data['Dates'].max().date()
    
    # Initialize session state for chart dates
    if 'chart_start_date' not in st.session_state:
        st.session_state.chart_start_date = max_date - timedelta(days=365)
    if 'chart_end_date' not in st.session_state:
        st.session_state.chart_end_date = max_date
    
    start_date = st.sidebar.date_input("Start Date", value=st.session_state.chart_start_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", value=st.session_state.chart_end_date, min_value=min_date, max_value=max_date)
    
    # Update session state
    st.session_state.chart_start_date = start_date
    st.session_state.chart_end_date = end_date
    
    # Fund filters
    available_asset_classes = etf_dict['Asset Class'].dropna().unique() if 'Asset Class' in etf_dict.columns else []
    available_geographies = etf_dict['Geografia'].dropna().unique() if 'Geografia' in etf_dict.columns else []
    
    selected_asset_classes = st.sidebar.multiselect("Asset Classes", available_asset_classes, default=available_asset_classes)
    selected_geographies = st.sidebar.multiselect("Geographies", available_geographies, default=available_geographies)
    
    # Custom scoring weights
    st.sidebar.markdown("## Scoring Weights")
    weights = {
        'YTD Return (%)': st.sidebar.slider("YTD Return", 0.0, 1.0, 0.3),
        '1Y Return (%)': st.sidebar.slider("1Y Return", 0.0, 1.0, 0.3),
        '2024 Return (%)': st.sidebar.slider("2024 Return", 0.0, 1.0, 0.2),
        'Volatility (%)': st.sidebar.slider("Low Volatility", 0.0, 1.0, 0.1),
        'Max Drawdown (%)': st.sidebar.slider("Low Drawdown", 0.0, 1.0, 0.1)
    }
    
    # Filter funds based on criteria
    filtered_funds = fund_columns
    
    if selected_asset_classes:
        asset_class_funds = etf_dict[etf_dict['Asset Class'].isin(selected_asset_classes)]['Ticker'].tolist()
        filtered_funds = [f for f in filtered_funds if f in asset_class_funds]
    
    if selected_geographies:
        geography_funds = etf_dict[etf_dict['Geografia'].isin(selected_geographies)]['Ticker'].tolist()
        filtered_funds = [f for f in filtered_funds if f in geography_funds]
    
    # Calculate performance metrics
    with st.spinner("Calculating performance metrics..."):
        performance_data = []
        
        for ticker in filtered_funds:
            if ticker in funds_data.columns:
                # Get fund info
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                
                if not fund_info.empty:
                    row = {
                        'Ticker': ticker,
                        'Fund Name': fund_info['Fund Name'].iloc[0] if 'Fund Name' in fund_info.columns else ticker,
                        'Asset Class': fund_info['Asset Class'].iloc[0] if 'Asset Class' in fund_info.columns else 'N/A',
                        'Geografia': fund_info['Geografia'].iloc[0] if 'Geografia' in fund_info.columns else 'N/A'
                    }
                    
                    # Calculate metrics
                    metrics = calculate_fund_metrics(funds_data, ticker)
                    if metrics:
                        row.update(metrics)
                        performance_data.append(row)
        
        if not performance_data:
            st.error("No funds meet the performance criteria.")
            return
        
        df_performance = pd.DataFrame(performance_data)
    
    # Calculate custom score
    df_scored = calculate_custom_score(df_performance, weights)
    
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
    
    # Show complete data table first
    st.markdown("## Fund Analysis Results")
    final_cols = ['Ticker', 'Fund Name', 'Custom Score'] + percentage_cols
    final_cols = [col for col in final_cols if col in display_df.columns]
    st.dataframe(display_df[final_cols], use_container_width=True)
    
    # Fund rankings with portfolio selection
    st.markdown("## Fund Rankings & Portfolio Selection")
    st.markdown(f"Select funds to add to your portfolio. Showing top {min(20, len(df_scored))} funds ordered by custom score.")
    
    # Show funds with selection buttons
    for idx, row in display_df.head(20).iterrows():
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Fund information
                score_text = f"**Score: {row['Custom Score']}**" if 'Custom Score' in row else ""
                st.markdown(f"**{row['Fund Name']}** ({row['Ticker']}) {score_text}")
                
                # Show key metrics
                metrics_text = []
                if 'YTD Return (%)' in row:
                    metrics_text.append(f"YTD: {row['YTD Return (%)']}")
                if '1Y Return (%)' in row:
                    metrics_text.append(f"1Y: {row['1Y Return (%)']}")
                if 'Volatility (%)' in row:
                    metrics_text.append(f"Vol: {row['Volatility (%)']}")
                
                if metrics_text:
                    st.caption(" | ".join(metrics_text))
            
            with col2:
                # Get display name for button
                display_name = row['Fund Name']
                fund_info = etf_dict[etf_dict['Ticker'] == row['Ticker']]
                if not fund_info.empty:
                    if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                        display_name = fund_info['Indice'].iloc[0]
                    elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                        display_name = fund_info['Fund Name'].iloc[0]
                
                PortfolioManager.render_fund_selector(
                    row['Ticker'], 
                    display_name, 
                    key_suffix=f"ranking_{idx}"
                )
            
            st.markdown("---")
    
    # Chart analysis
    st.markdown("## Chart Analysis")
    
    # Fund selector for chart
    top_funds = list(df_scored.head(10)['Ticker'])
    fund_names_for_chart = {}
    for ticker in top_funds:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        fund_name = ticker  # Default
        if not fund_info.empty:
            if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                fund_name = fund_info['Indice'].iloc[0]
            elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                fund_name = fund_info['Fund Name'].iloc[0]
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
        
        # Create and display chart
        chart = create_cumulative_returns_chart(
            funds_data, 
            selected_funds, 
            pd.to_datetime(st.session_state.chart_start_date), 
            pd.to_datetime(st.session_state.chart_end_date),
            fund_names_for_chart
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    else:
        st.info("Select funds from the list above to view cumulative returns chart.")

def render_portfolio_analysis():
    """Render portfolio analysis page"""
    PortfolioManager.initialize()
    
    # Back button
    if st.button("← Back to Main Dashboard"):
        st.session_state.show_portfolio_analysis = False
        st.rerun()
    
    st.markdown("# Portfolio Analysis")
    
    if not st.session_state.selected_funds:
        st.info("No assets selected. Go to the main dashboard to select funds.")
        return
    
    # Load data
    funds_data, etf_dict = load_data()
    if funds_data is None or etf_dict is None:
        st.error("Error loading data")
        return
    
    # Portfolio summary
    st.markdown("## Portfolio Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Selected Assets", len(st.session_state.selected_funds))
    with col2:
        total_weight = sum(st.session_state.portfolio_weights.values())
        st.metric("Total Allocation", f"{total_weight:.1f}%")
    
    # Show individual fund metrics
    st.markdown("## Individual Fund Analysis")
    
    individual_metrics = []
    for ticker in st.session_state.selected_funds:
        metrics = calculate_fund_metrics(funds_data, ticker)
        if metrics:
            fund_info = etf_dict[etf_dict['Ticker'] == ticker]
            
            # Get fund name
            fund_name = ticker
            if not fund_info.empty:
                if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                    fund_name = fund_info['Indice'].iloc[0]
                elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                    fund_name = fund_info['Fund Name'].iloc[0]
            
            metrics['Fund'] = fund_name
            metrics['Ticker'] = ticker
            metrics['Weight (%)'] = st.session_state.portfolio_weights.get(ticker, 0)
            individual_metrics.append(metrics)
    
    if individual_metrics:
        metrics_df = pd.DataFrame(individual_metrics)
        
        # Reorder columns
        cols = ['Fund', 'Ticker', 'Weight (%)'] + [col for col in metrics_df.columns if col not in ['Fund', 'Ticker', 'Weight (%)']]
        metrics_df = metrics_df[cols]
        
        st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Weight Management")
    st.markdown("Adjust individual fund weights:")
    
    # Weight adjustment
    funds_to_remove = []
    
    for ticker in st.session_state.selected_funds:
        current_weight = st.session_state.portfolio_weights.get(ticker, 0)
        
        # Get fund info
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        fund_name = ticker
        
        if not fund_info.empty:
            if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                fund_name = fund_info['Indice'].iloc[0]
            elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                fund_name = fund_info['Fund Name'].iloc[0]
        
        col_name, col_weight, col_remove = st.columns([3, 1.5, 1])
        
        with col_name:
            st.markdown(f"**{fund_name}**")
            st.caption(f"Ticker: {ticker}")
        
        with col_weight:
            new_weight = st.number_input(
                "Weight %",
                min_value=0.0,
                max_value=100.0,
                value=float(current_weight),
                step=0.5,
                key=f"weight_{ticker}",
                label_visibility="collapsed"
            )
            st.session_state.portfolio_weights[ticker] = new_weight
        
        with col_remove:
            st.write("")  # Spacing
            if st.button("🗑️", key=f"remove_{ticker}", help="Remove from portfolio"):
                funds_to_remove.append(ticker)
    
    # Process removals
    for ticker in funds_to_remove:
        st.session_state.selected_funds.discard(ticker)
        if ticker in st.session_state.portfolio_weights:
            del st.session_state.portfolio_weights[ticker]
        st.rerun()
    
    # Rebalance button
    if st.button("⚖️ Equal Weight Rebalance"):
        num_funds = len(st.session_state.selected_funds)
        if num_funds > 0:
            equal_weight = 100.0 / num_funds
            for ticker in st.session_state.selected_funds:
                st.session_state.portfolio_weights[ticker] = equal_weight
            st.rerun()

if __name__ == "__main__":
    main()