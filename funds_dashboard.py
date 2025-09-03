#!/usr/bin/env python3
"""
FUNDS ANALYSIS DASHBOARD
Professional fund analysis with interactive filtering and performance metrics
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

# Page configuration
st.set_page_config(
    page_title="Dashboard de Análisis de Fondos",
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
    
    /* Dark theme toggle button */
    .theme-toggle {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
        background: #262730;
        border: 1px solid #404040;
        border-radius: 5px;
        padding: 5px 10px;
        color: #fafafa;
        cursor: pointer;
    }
    
    /* Improve dataframe styling in dark mode */
    .stDataFrame {
        background-color: #262730;
    }
    
    /* Better contrast for charts */
    .js-plotly-plot {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load funds data and dictionary from CSV files"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Load price data
        funds_path = os.path.join(data_dir, 'funds_prices.csv')
        funds = pd.read_csv(funds_path)
        
        # Load ETF dictionary
        dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
        etf_dict = pd.read_csv(dict_path)
        
        return funds, etf_dict
        
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.error("Please check that the following files exist:")
        st.error(f"- {os.path.join(script_dir, 'data', 'funds_prices.csv')}")
        st.error(f"- {os.path.join(script_dir, 'data', 'funds_dictionary.csv')}")
        st.error("Run 'python convert_data.py' first to generate the CSV files.")
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
        
        # Get fund prices and calculate returns
        prices = funds_df[['Dates', fund_ticker]].dropna()
        if len(prices) < 2:
            return None
        
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        
        # Calculate returns
        prices['Returns'] = prices[fund_ticker].pct_change()
        
        # Current date for calculations
        current_date = prices['Dates'].max()
        current_year = current_date.year
        
        # YTD Return
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        if len(ytd_data) > 1:
            ytd_return = ((ytd_data[fund_ticker].iloc[-1] / ytd_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            ytd_return = 0
        
        # MTD Return (Month-to-Date)
        mtd_start = pd.to_datetime(f'{current_date.year}-{current_date.month:02d}-01')
        mtd_data = prices[prices['Dates'] >= mtd_start]
        if len(mtd_data) > 1:
            mtd_return = ((mtd_data[fund_ticker].iloc[-1] / mtd_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            mtd_return = 0
        
        # Monthly Return (last 30 days)
        monthly_start = current_date - timedelta(days=30)
        monthly_data = prices[prices['Dates'] >= monthly_start]
        if len(monthly_data) > 1:
            monthly_return = ((monthly_data[fund_ticker].iloc[-1] / monthly_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            monthly_return = 0
        
        # 1 Year Return
        year_1_start = current_date - timedelta(days=365)
        year_1_data = prices[prices['Dates'] >= year_1_start]
        if len(year_1_data) > 1:
            return_1y = ((year_1_data[fund_ticker].iloc[-1] / year_1_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_1y = 0
        
        # 2024 Return
        year_2024_start = pd.to_datetime('2024-01-01')
        year_2024_end = pd.to_datetime('2024-12-31')
        year_2024_data = prices[(prices['Dates'] >= year_2024_start) & (prices['Dates'] <= year_2024_end)]
        if len(year_2024_data) > 1:
            return_2024 = ((year_2024_data[fund_ticker].iloc[-1] / year_2024_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2024 = 0
        
        # 2023 Return
        year_2023_start = pd.to_datetime('2023-01-01')
        year_2023_end = pd.to_datetime('2023-12-31')
        year_2023_data = prices[(prices['Dates'] >= year_2023_start) & (prices['Dates'] <= year_2023_end)]
        if len(year_2023_data) > 1:
            return_2023 = ((year_2023_data[fund_ticker].iloc[-1] / year_2023_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2023 = 0
        
        # 2022 Return
        year_2022_start = pd.to_datetime('2022-01-01')
        year_2022_end = pd.to_datetime('2022-12-31')
        year_2022_data = prices[(prices['Dates'] >= year_2022_start) & (prices['Dates'] <= year_2022_end)]
        if len(year_2022_data) > 1:
            return_2022 = ((year_2022_data[fund_ticker].iloc[-1] / year_2022_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2022 = 0
        
        # Max Drawdown
        cumulative = (1 + prices['Returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Volatility (annualized)
        volatility = prices['Returns'].std() * np.sqrt(252) * 100
        
        # VaR and CVaR at 5% confidence level (daily, then annualized)
        daily_returns = prices['Returns'].dropna()
        var_5_annual = 0
        cvar_5_annual = 0
        
        if len(daily_returns) >= 20:  # Need sufficient data for VaR
            # Sort returns for percentile calculation
            sorted_returns = np.sort(daily_returns)
            
            # VaR at 5% (5th percentile of losses)
            var_5_daily = np.percentile(sorted_returns, 5)
            var_5_annual = var_5_daily * np.sqrt(252) * 100  # Annualized percentage
            
            # CVaR at 5% (expected shortfall - average of worst 5% returns)
            var_index = int(0.05 * len(sorted_returns))
            if var_index > 0:
                cvar_5_daily = sorted_returns[:var_index].mean()
                cvar_5_annual = cvar_5_daily * np.sqrt(252) * 100  # Annualized percentage
        
        return {
            'YTD Return (%)': ytd_return,
            'MTD Return (%)': mtd_return,
            'Monthly Return (%)': monthly_return,
            '1Y Return (%)': return_1y,
            '2024 Return (%)': return_2024,
            '2023 Return (%)': return_2023,
            '2022 Return (%)': return_2022,
            'Max Drawdown (%)': max_drawdown,
            'Volatility (%)': volatility,
            'VaR 5% (%)': var_5_annual,
            'CVaR 5% (%)': cvar_5_annual
        }
        
    except Exception as e:
        st.error(f"Error calculating metrics for {fund_ticker}: {e}")
        return None

def calculate_custom_score(df_performance, weights):
    """Calculate custom score based on user-defined weights using Z-score standardization"""
    try:
        df_scored = df_performance.copy()
        
        # Define metrics and their direction for scoring
        # IMPORTANT: Max Drawdown, VaR, CVaR are already NEGATIVE values from calculation
        # So we need to handle them correctly to avoid double inversion
        metrics_to_score = {
            'YTD Return (%)': 'positive',           # Higher return = better
            'MTD Return (%)': 'positive',           # Higher return = better
            'Monthly Return (%)': 'positive',       # Higher return = better
            '1Y Return (%)': 'positive',            # Higher return = better
            '2024 Return (%)': 'positive',          # Higher return = better
            '2023 Return (%)': 'positive',          # Higher return = better
            '2022 Return (%)': 'positive',          # Higher return = better
            'Max Drawdown (%)': 'negative_value',   # Already negative, higher (less negative) = better
            'Volatility (%)': 'negative',           # Positive value, lower = better
            'VaR 5% (%)': 'negative_value',         # Already negative, higher (less negative) = better
            'CVaR 5% (%)': 'negative_value'         # Already negative, higher (less negative) = better
        }
        
        # Calculate Z-scores for each metric
        z_scores = {}
        for metric, direction in metrics_to_score.items():
            if metric in df_scored.columns:
                metric_values = df_scored[metric].dropna()
                
                if len(metric_values) > 1:
                    mean_val = metric_values.mean()
                    std_val = metric_values.std()
                    
                    if std_val > 0:
                        # Calculate Z-score: (value - mean) / std
                        z_score = (df_scored[metric] - mean_val) / std_val
                        
                        # Handle different metric types correctly
                        if direction == 'negative':
                            # For positive values where lower is better (e.g., Volatility)
                            z_score = -z_score
                        elif direction == 'negative_value':
                            # For already negative values where higher (less negative) is better
                            # (e.g., Max Drawdown: -10% is better than -20%)
                            # Don't invert - higher Z-score already means less negative (better)
                            pass  # Keep z_score as is
                        # For 'positive' direction, keep z_score as is (higher = better)
                        
                        z_scores[metric] = z_score
                    else:
                        # If no variation, assign neutral score (0)
                        z_scores[metric] = pd.Series([0] * len(df_scored), index=df_scored.index)
                else:
                    # If only one value, assign neutral score
                    z_scores[metric] = pd.Series([0] * len(df_scored), index=df_scored.index)
        
        # Calculate weighted composite score
        scores = []
        
        for idx, row in df_scored.iterrows():
            total_score = 0
            total_weight = 0
            
            for metric in metrics_to_score.keys():
                if metric in weights and weights[metric] > 0 and metric in z_scores:
                    weight = weights[metric] / 100  # Convert percentage to decimal
                    z_value = z_scores[metric].loc[idx]
                    
                    if pd.notnull(z_value):
                        total_score += z_value * weight
                        total_weight += weight
            
            # Calculate final score (weighted average of Z-scores)
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0  # Neutral score if no weights
            
            scores.append(final_score)
        
        df_scored['Custom Score'] = scores
        
        # Sort by score (higher is better)
        return df_scored.sort_values('Custom Score', ascending=False)
        
    except Exception as e:
        st.error(f"Error calculating custom score: {e}")
        return df_performance

def create_cumulative_returns_chart(funds_df, selected_funds, start_date, end_date):
    """Create interactive cumulative returns chart with custom date range"""
    try:
        # Filter data by date range
        funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
        filtered_data = funds_df[(funds_df['Dates'] >= start_date) & (funds_df['Dates'] <= end_date)].copy()
        
        if filtered_data.empty:
            return None
        
        fig = go.Figure()
        
        # Dark theme optimized colors - more vibrant and contrasting
        colors = ['#60a5fa', '#f87171', '#34d399', '#fbbf24', '#a78bfa', 
                 '#22d3ee', '#fb923c', '#a3e635', '#f472b6', '#818cf8']
        
        for i, fund in enumerate(selected_funds):
            if fund in filtered_data.columns:
                # Calculate cumulative returns (base 100)
                prices = filtered_data[['Dates', fund]].dropna()
                if len(prices) > 1:
                    base_price = prices[fund].iloc[0]
                    cumulative_returns = (prices[fund] / base_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=prices['Dates'],
                        y=cumulative_returns,
                        mode='lines',
                        name=fund,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Value: %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ))
        
        fig.update_layout(
            title=f"Cumulative Returns Evolution (Base 100)<br><sub>From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</sub>",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Base 100)",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            # Dark theme optimized layout
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            title_font=dict(color='#fafafa', size=16),
            xaxis=dict(
                gridcolor='#404040',
                zerolinecolor='#404040',
                color='#fafafa'
            ),
            yaxis=dict(
                gridcolor='#404040',
                zerolinecolor='#404040',
                color='#fafafa'
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def calculate_efficient_frontier_cvxpy(funds_df, fund_tickers, debug_container=None):
    """
    Calculate efficient frontier using CVXPY optimization - EXACT same logic as codigo_ignacio.py
    """
    def debug_log(message):
        if debug_container:
            debug_container.write(message)
        else:
            print(message)
    
    try:
        # Validate inputs
        if len(fund_tickers) < 2:
            debug_log(f"❌ No hay suficientes fondos: {len(fund_tickers)} (mínimo 2)")
            return None
        
        # Limit to maximum 10 funds for computational efficiency
        if len(fund_tickers) > 10:
            fund_tickers = fund_tickers[:10]
            debug_log(f"⚠️ Limitado a los primeros 10 fondos para eficiencia")
            
        # Filter data for selected funds
        available_tickers = [ticker for ticker in fund_tickers if ticker in funds_df.columns]
        if len(available_tickers) < 2:
            debug_log(f"❌ Fondos disponibles insuficientes: {len(available_tickers)} de {len(fund_tickers)}")
            debug_log(f"Fondos solicitados: {fund_tickers[:5]}...")
            debug_log(f"Fondos disponibles: {available_tickers}")
            return None
            
        funds_data = funds_df[['Dates'] + available_tickers].copy()
        funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
        funds_data = funds_data.dropna()
        
        if len(funds_data) < 50:
            debug_log(f"❌ Datos insuficientes: {len(funds_data)} observaciones (mínimo 50)")
            return None
        
        # Prepare data - EXACT same as Ignacio's code
        funds_data = funds_data.set_index('Dates')
        returns = funds_data.pct_change().dropna()
        
        # === EXACT same logic as codigo_ignacio.py ===
        mu_ann = returns.mean() * 252
        cov_ann = returns.cov() * 252
        
        n = len(available_tickers)
        debug_log(f"✅ Procesando {n} fondos con {len(returns)} observaciones de retornos")
        
        # Max Sharpe Portfolio - EXACT same as Ignacio's code
        w_sh = cp.Variable(n)
        ret_sh = mu_ann.values @ w_sh
        var_sh = cp.quad_form(w_sh, cov_ann.values)
        cons = [cp.sum(w_sh) == 1, ret_sh >= 1e-4, w_sh >= 0]
        
        # EXACT same as Ignacio's code - single line solve
        debug_log("🔧 Resolviendo optimización Max Sharpe...")
        cp.Problem(cp.Minimize(var_sh), cons).solve()
        
        if w_sh.value is None:
            debug_log("❌ Optimización Max Sharpe falló")
            return None
            
        w_opt = w_sh.value
        ret_opt = mu_ann.values @ w_opt
        std_opt = np.sqrt(w_opt.T @ cov_ann.values @ w_opt)
        debug_log(f"✅ Max Sharpe calculado: Retorno={ret_opt:.4f}, Volatilidad={std_opt:.4f}")
        
        # Generate efficient frontier - EXACT same as Ignacio's code
        targets, data, frontier = np.linspace(mu_ann.min(), mu_ann.max(), 50), [], []
        successful_points = 0
        
        debug_log(f"🔧 Generando frontera eficiente con {len(targets)} puntos...")
        
        for i, t in enumerate(targets):
            w = cp.Variable(n)
            risk = cp.quad_form(w, cov_ann.values)
            c = [cp.sum(w) == 1, mu_ann.values @ w == t, w >= 0]
            try:
                # EXACT same as Ignacio's code - single line solve
                cp.Problem(cp.Minimize(risk), c).solve()
                
                if w.value is not None and risk.value is not None:
                    std = np.sqrt(risk.value)
                    sr = t / std if std > 0 else 0
                    # EXACT same structure as Ignacio's code
                    row = {asset: weight for asset, weight in zip(available_tickers, w.value)}
                    row.update({
                        "Retorno esperado": t,
                        "Volatilidad": std,
                        "Sharpe Ratio": sr
                    })
                    data.append(row)
                    frontier.append((std, t, sr))
                    successful_points += 1
            except:
                # EXACT same as Ignacio's code - simple continue
                continue
        
        debug_log(f"📈 Generados {successful_points} puntos válidos de {len(targets)} objetivos")
        
        if len(data) == 0:
            debug_log("❌ No se generaron puntos válidos de frontera")
            return None
        
        # Create DataFrame with frontier data
        frontier_df = pd.DataFrame(data)
        
        # Return data in format expected by chart function
        return {
            'frontier_points': frontier,
            'max_sharpe': (std_opt, ret_opt),
            'individual_assets': [(returns[asset].std() * np.sqrt(252), returns[asset].mean() * 252, asset) for asset in available_tickers],
            'frontier_df': frontier_df
        }
        
    except Exception as e:
        print(f"CVXPY efficient frontier calculation failed: {e}")
        return None

def create_efficient_frontier_chart(funds_df, fund_tickers, fund_names_dict, df_performance, debug_container=None):
    """Create efficient frontier chart - EXACT same logic as codigo_ignacio.py"""
    try:
        # Calculate efficient frontier with CVXPY - same as Ignacio's code
        result = calculate_efficient_frontier_cvxpy(funds_df, fund_tickers, debug_container)
        
        if result is None:
            if debug_container:
                debug_container.write("❌ calculate_efficient_frontier_cvxpy retornó None")
            return None, None
        
        # Extract data from result
        frontier_points = result['frontier_points']
        max_sharpe = result['max_sharpe']
        individual_assets = result['individual_assets']
        frontier_df = result['frontier_df']
        
        # Create figure - EXACT same as Ignacio's code
        fig_f = go.Figure()
        
        # Add frontier points - dark theme optimized
        for s, r, sr in frontier_points:
            fig_f.add_trace(go.Scatter(x=[s], y=[r], mode='markers',
                marker=dict(size=6, color='#60a5fa'), showlegend=False))
        
        # Add Max Sharpe point - dark theme optimized
        std_opt, ret_opt = max_sharpe
        fig_f.add_trace(go.Scatter(x=[std_opt], y=[ret_opt], mode='markers',
            marker=dict(size=10, color='#f87171'), name='Max Sharpe'))
        
        # Add individual assets - only markers, names on hover
        for s_a, r_a, asset_name in individual_assets:
            # Get display name
            display_name = fund_names_dict.get(asset_name, asset_name)
            fig_f.add_trace(go.Scatter(x=[s_a], y=[r_a], mode='markers',
                name=display_name, showlegend=False,
                marker=dict(size=8, color='#9ca3af'),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Volatilidad: %{x:.2%}<br>' +
                             'Retorno: %{y:.2%}<br>' +
                             '<extra></extra>'))
        
        # Update layout - dark theme optimized
        fig_f.update_layout(
            title="Frontera Eficiente",
            xaxis_title="Volatilidad Anual",
            yaxis_title="Retorno Anual",
            height=600,
            showlegend=True,
            xaxis=dict(
                tickformat='.1%',
                gridcolor='#404040',
                zerolinecolor='#404040',
                color='#fafafa'
            ),
            yaxis=dict(
                tickformat='.1%',
                gridcolor='#404040',
                zerolinecolor='#404040',
                color='#fafafa'
            ),
            # Dark theme layout
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            title_font=dict(color='#fafafa', size=16)
        )
        
        return fig_f, frontier_df
        
    except Exception as e:
        st.error(f"Error creating efficient frontier chart: {e}")
        return None, None

def create_excel_report(df_performance, chart_fig, frontier_fig, weights, filtered_funds_count, funds_df=None, selected_funds=None, frontier_df=None):
    """Create comprehensive Excel report with all data including efficient frontier and charts"""
    try:
        output = io.BytesIO()
        
        # Use xlsxwriter for better chart support
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Main performance data (complete analysis)
            df_performance.to_excel(writer, sheet_name='Fund Analysis', index=False)
            
            # Efficient Frontier data (if available)
            if frontier_df is not None and not frontier_df.empty:
                # Round numerical columns for better readability
                frontier_export = frontier_df.copy()
                numerical_cols = frontier_export.select_dtypes(include=[np.number]).columns
                frontier_export[numerical_cols] = frontier_export[numerical_cols].round(4)
                frontier_export.to_excel(writer, sheet_name='Efficient Frontier', index=False)
            
            # Weights configuration
            weights_df = pd.DataFrame(list(weights.items()), columns=['Metric', 'Weight (%)'])
            weights_df.to_excel(writer, sheet_name='Scoring Weights', index=False)
            
            # Summary statistics
            summary_data = {
                'Report Generated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Total Funds Analyzed': [filtered_funds_count],
                'Total Weight': [f"{sum(weights.values())}%"],
                'Top Ranked Fund': [df_performance.iloc[0]['Fund Name'] if not df_performance.empty else 'N/A'],
                'Top Custom Score': [f"{df_performance.iloc[0]['Custom Score']:.2f}" if not df_performance.empty and 'Custom Score' in df_performance.columns else 'N/A'],
                'Analysis Date Range': [f"{funds_df['Dates'].min().date()} to {funds_df['Dates'].max().date()}" if funds_df is not None else 'N/A'],
                'Total Data Points': [len(funds_df) if funds_df is not None else 'N/A'],
                'Efficient Frontier Points': [len(frontier_df) if frontier_df is not None else 'N/A']
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Report Summary', index=False)
            
            # Risk metrics summary
            if not df_performance.empty:
                risk_cols = ['Fund Name', 'Ticker', 'Region', 'Asset Class', 'Sector', 'Volatility (%)', 'Max Drawdown (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
                available_risk_cols = [col for col in risk_cols if col in df_performance.columns]
                risk_df = df_performance[available_risk_cols].copy()
                risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # Performance metrics breakdown
            if not df_performance.empty:
                perf_cols = ['Fund Name', 'Ticker', 'YTD Return (%)', 'MTD Return (%)', 'Monthly Return (%)', 
                           '1Y Return (%)', '2024 Return (%)', '2023 Return (%)', '2022 Return (%)']
                available_perf_cols = [col for col in perf_cols if col in df_performance.columns]
                perf_df = df_performance[available_perf_cols].copy()
                perf_df.to_excel(writer, sheet_name='Returns Analysis', index=False)
            
            # Raw price data for selected funds (if available)
            if funds_df is not None and selected_funds is not None and len(selected_funds) > 0:
                # Get available tickers from selected funds
                available_tickers = [ticker for ticker in selected_funds if ticker in funds_df.columns]
                if available_tickers:
                    price_cols = ['Dates'] + available_tickers[:10]  # Limit to 10 funds to avoid Excel limits
                    price_data = funds_df[price_cols].copy()
                    price_data.to_excel(writer, sheet_name='Price Data (Selected)', index=False)
            
            # Top performers by category
            if not df_performance.empty and 'Custom Score' in df_performance.columns:
                top_10 = df_performance.head(10)[['Fund Name', 'Ticker', 'Custom Score', 'YTD Return (%)', 'Volatility (%)']]
                top_10.to_excel(writer, sheet_name='Top 10 Funds', index=False)
            
            # Add charts as images if available
            workbook = writer.book
            
            if chart_fig is not None:
                try:
                    # Create a new worksheet for charts
                    chart_worksheet = workbook.add_worksheet('Charts')
                    
                    # Convert chart to image and add to Excel
                    chart_img_bytes = chart_fig.to_image(format="png", width=800, height=600)
                    chart_worksheet.insert_image('A1', '', {'image_data': chart_img_bytes})
                    
                    # Add title
                    title_format = workbook.add_format({'bold': True, 'font_size': 14})
                    chart_worksheet.write('A40', 'Cumulative Returns Chart', title_format)
                    
                except Exception as e:
                    print(f"Could not add cumulative returns chart to Excel: {e}")
            
            if frontier_fig is not None:
                try:
                    # Add frontier chart to the same worksheet or create new one
                    if 'chart_worksheet' not in locals():
                        chart_worksheet = workbook.add_worksheet('Charts')
                    
                    # Convert frontier chart to image and add to Excel
                    frontier_img_bytes = frontier_fig.to_image(format="png", width=800, height=600)
                    chart_worksheet.insert_image('A45', '', {'image_data': frontier_img_bytes})
                    
                    # Add title
                    title_format = workbook.add_format({'bold': True, 'font_size': 14})
                    chart_worksheet.write('A85', 'Efficient Frontier Chart', title_format)
                    
                except Exception as e:
                    print(f"Could not add efficient frontier chart to Excel: {e}")
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error creating Excel report: {e}")
        return None

def create_csv_report(df_performance):
    """Create CSV report"""
    try:
        output = io.StringIO()
        df_performance.to_csv(output, index=False)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error creating CSV report: {e}")
        return None

def fig_to_base64(fig):
    """Convert plotly figure to base64 string for PDF"""
    try:
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode()
        return img_base64
    except Exception as e:
        st.error(f"Error converting figure to base64: {e}")
        return None

def create_pdf_report(df_performance, chart_fig, frontier_fig, weights, filtered_funds_count, frontier_df=None):
    """Create comprehensive PDF report including efficient frontier analysis"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        story.append(Paragraph("Reporte de Análisis de Fondos", title_style))
        story.append(Spacer(1, 20))
        
        # Report summary
        summary_style = styles['Heading2']
        story.append(Paragraph("Resumen del Reporte", summary_style))
        
        summary_data = [
            ['Reporte Generado', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Fondos Analizados', str(filtered_funds_count)],
            ['Peso Total Scoring', f"{sum(weights.values())}%"],
            ['Fondo Top Ranking', df_performance.iloc[0]['Fund Name'] if not df_performance.empty else 'N/A'],
            ['Score Personalizado Top', f"{df_performance.iloc[0]['Custom Score']:.1f}" if not df_performance.empty and 'Custom Score' in df_performance.columns else 'N/A'],
            ['Puntos Frontera Eficiente', str(len(frontier_df)) if frontier_df is not None else 'N/A']
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Scoring weights
        story.append(Paragraph("Pesos de Scoring Personalizado", summary_style))
        weights_data = [['Métrica', 'Peso (%)']]
        for metric, weight in weights.items():
            weights_data.append([metric, f"{weight}%"])
        
        weights_table = Table(weights_data)
        weights_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(weights_table)
        story.append(Spacer(1, 20))
        
        # Top 10 funds table
        if not df_performance.empty:
            story.append(Paragraph("Top 10 Ranked Funds", summary_style))
            
            # Select key columns for the PDF table
            key_cols = ['Fund Name', 'Custom Score', 'YTD Return (%)', '1Y Return (%)', 'Volatility (%)', 'Max Drawdown (%)']
            top_10 = df_performance.head(10)[key_cols].copy()
            
            # Format the data for table
            table_data = [key_cols]  # Header
            for _, row in top_10.iterrows():
                formatted_row = []
                for col in key_cols:
                    if col == 'Fund Name':
                        # Truncate long names
                        name = str(row[col])
                        formatted_row.append(name[:30] + "..." if len(name) > 30 else name)
                    elif col == 'Custom Score':
                        formatted_row.append(f"{row[col]:.1f}" if pd.notnull(row[col]) else "N/A")
                    else:
                        formatted_row.append(f"{row[col]:.2f}%" if pd.notnull(row[col]) else "N/A")
                table_data.append(formatted_row)
            
            funds_table = Table(table_data)
            funds_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(funds_table)
        
        # Efficient Frontier Analysis (if available)
        if frontier_df is not None and not frontier_df.empty:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Análisis de Frontera Eficiente", summary_style))
            
            # Get optimal portfolios from frontier
            max_sharpe_idx = frontier_df['Sharpe Ratio'].idxmax()
            min_vol_idx = frontier_df['Volatilidad'].idxmin()
            
            frontier_summary = [
                ['Métrica', 'Portfolio Max Sharpe', 'Portfolio Min Volatilidad'],
                ['Retorno Esperado', f"{frontier_df.loc[max_sharpe_idx, 'Retorno esperado']:.2%}", f"{frontier_df.loc[min_vol_idx, 'Retorno esperado']:.2%}"],
                ['Volatilidad', f"{frontier_df.loc[max_sharpe_idx, 'Volatilidad']:.2%}", f"{frontier_df.loc[min_vol_idx, 'Volatilidad']:.2%}"],
                ['Sharpe Ratio', f"{frontier_df.loc[max_sharpe_idx, 'Sharpe Ratio']:.3f}", f"{frontier_df.loc[min_vol_idx, 'Sharpe Ratio']:.3f}"]
            ]
            
            frontier_table = Table(frontier_summary)
            frontier_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(frontier_table)
        
        # Note about charts
        story.append(Spacer(1, 20))
        story.append(Paragraph("Nota: Los gráficos interactivos están disponibles en el dashboard web. Este PDF contiene únicamente los datos tabulares.", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating PDF report: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Theme toggle in sidebar
    st.sidebar.markdown("### 🎨 Tema")
    theme_choice = st.sidebar.selectbox(
        "Seleccionar tema:",
        ["🌙 Modo Oscuro", "☀️ Modo Claro"],
        index=0  # Default to dark mode
    )
    
    # Apply theme-specific styling
    if theme_choice == "☀️ Modo Claro":
        st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff;
                color: #1f2937;
            }
            .main-header {
                color: #1f2937 !important;
            }
            .metric-card {
                background-color: #f9fafb !important;
                border: 1px solid #e5e7eb !important;
            }
            .positive-return {
                color: #059669 !important;
            }
            .negative-return {
                color: #dc2626 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">📈 Dashboard de Análisis de Fondos</h1>', unsafe_allow_html=True)
    
    # Load data
    funds_data, etf_dict = load_data()
    
    if funds_data is None or etf_dict is None:
        st.stop()
    
    # Get available funds
    available_funds = [col for col in funds_data.columns if col != 'Dates']
    
    # SIDEBAR - Filters
    st.sidebar.markdown("# 🔍 Filters")
    
    # Create mapping from ETF dictionary
    fund_info = {}
    if not etf_dict.empty:
        for _, row in etf_dict.iterrows():
            dict_ticker = row.get('Ticker', '')
            
            # Try to match with available funds
            matched_fund = None
            for fund in available_funds:
                if fund == dict_ticker:
                    matched_fund = fund
                    break
                if fund.replace(' ', '').upper() == dict_ticker.replace(' ', '').upper():
                    matched_fund = fund
                    break
                if dict_ticker.replace(' ', '').upper() in fund.replace(' ', '').upper():
                    matched_fund = fund
                    break
            
            if matched_fund:
                fund_info[matched_fund] = {
                    'name': row.get('Indice', matched_fund),
                    'region': row.get('Geografia', 'Unknown'),
                    'asset_class': row.get('Asset Class', 'Unknown'),
                    'subclass': row.get('Subclass', 'Unknown'),
                    'sector': row.get('Sector', 'Unknown'),
                    'original_ticker': dict_ticker
                }
    
    # Get unique categories for filters (all available initially)
    all_regions = list(set([info.get('region', 'Unknown') for info in fund_info.values()]))
    all_asset_classes = list(set([info.get('asset_class', 'Unknown') for info in fund_info.values()]))
    all_subclasses = list(set([info.get('subclass', 'Unknown') for info in fund_info.values()]))
    all_sectors = list(set([info.get('sector', 'Unknown') for info in fund_info.values()]))
    
    # Remove 'Unknown' if there are other options
    if len(all_regions) > 1 and 'Unknown' in all_regions:
        all_regions.remove('Unknown')
    if len(all_asset_classes) > 1 and 'Unknown' in all_asset_classes:
        all_asset_classes.remove('Unknown')
    if len(all_subclasses) > 1 and 'Unknown' in all_subclasses:
        all_subclasses.remove('Unknown')
    if len(all_sectors) > 1 and 'Unknown' in all_sectors:
        all_sectors.remove('Unknown')
    
    # Data summary
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.write(f"📈 Total funds: {len(available_funds)}")
    st.sidebar.write(f"🏷️ With metadata: {len(fund_info)}")
    st.sidebar.write(f"🌍 Regions: {len(all_regions)}")
    st.sidebar.write(f"💼 Asset classes: {len(all_asset_classes)}")
    st.sidebar.write(f"🏷️ Subclasses: {len(all_subclasses)}")
    st.sidebar.write(f"🏭 Sectors: {len(all_sectors)}")
    
    # Filter controls with cascading logic
    st.sidebar.markdown("### 🎯 Filter Controls")
    
    # Region filter (always shows all regions)
    selected_regions = st.sidebar.multiselect(
        "Seleccionar Regiones:",
        options=sorted(all_regions),
        default=[]  # Start with no selection
    )
    
    # Filter available asset classes based on selected regions
    if selected_regions:
        available_asset_classes = list(set([
            info.get('asset_class', 'Unknown') 
            for info in fund_info.values() 
            if info.get('region', 'Unknown') in selected_regions
        ]))
        if 'Unknown' in available_asset_classes and len(available_asset_classes) > 1:
            available_asset_classes.remove('Unknown')
    else:
        available_asset_classes = all_asset_classes
    
    # Asset class filter
    selected_asset_classes = st.sidebar.multiselect(
        "Seleccionar Clases de Activos:",
        options=sorted(available_asset_classes),
        default=[]  # Start with no selection
    )
    
    # Filter available subclasses based on selected regions and asset classes
    if selected_regions or selected_asset_classes:
        available_subclasses = list(set([
            info.get('subclass', 'Unknown') 
            for info in fund_info.values() 
            if (not selected_regions or info.get('region', 'Unknown') in selected_regions) and
               (not selected_asset_classes or info.get('asset_class', 'Unknown') in selected_asset_classes)
        ]))
        if 'Unknown' in available_subclasses and len(available_subclasses) > 1:
            available_subclasses.remove('Unknown')
    else:
        available_subclasses = all_subclasses
    
    # Subclass filter
    selected_subclasses = st.sidebar.multiselect(
        "Seleccionar Subclases:",
        options=sorted(available_subclasses),
        default=[]  # Start with no selection
    )
    
    # Filter available sectors based on all previous selections
    if selected_regions or selected_asset_classes or selected_subclasses:
        available_sectors = list(set([
            info.get('sector', 'Unknown') 
            for info in fund_info.values() 
            if (not selected_regions or info.get('region', 'Unknown') in selected_regions) and
               (not selected_asset_classes or info.get('asset_class', 'Unknown') in selected_asset_classes) and
               (not selected_subclasses or info.get('subclass', 'Unknown') in selected_subclasses)
        ]))
        if 'Unknown' in available_sectors and len(available_sectors) > 1:
            available_sectors.remove('Unknown')
    else:
        available_sectors = all_sectors
    
    # Sector filter
    selected_sectors = st.sidebar.multiselect(
        "Seleccionar Sectores:",
        options=sorted(available_sectors),
        default=[]  # Start with no selection
    )
    
    # Apply filters
    filtered_funds = []
    for fund in available_funds:
        if fund in fund_info:
            fund_region = fund_info[fund].get('region', 'Unknown')
            fund_asset_class = fund_info[fund].get('asset_class', 'Unknown')
            fund_subclass = fund_info[fund].get('subclass', 'Unknown')
            fund_sector = fund_info[fund].get('sector', 'Unknown')
            
            # Include fund if it matches selected filters OR if no filters are selected
            region_match = not selected_regions or fund_region in selected_regions
            asset_class_match = not selected_asset_classes or fund_asset_class in selected_asset_classes
            subclass_match = not selected_subclasses or fund_subclass in selected_subclasses
            sector_match = not selected_sectors or fund_sector in selected_sectors
            
            if region_match and asset_class_match and subclass_match and sector_match:
                filtered_funds.append(fund)
        else:
            # Include funds not in dictionary if no specific filters are applied
            if not selected_regions and not selected_asset_classes and not selected_subclasses and not selected_sectors:
                filtered_funds.append(fund)
    
    # Show filtered results
    st.sidebar.write(f"**Filtered funds: {len(filtered_funds)}**")
    
    # Custom Scoring System
    st.sidebar.markdown("### 🎯 Custom Fund Ranking")
    st.sidebar.markdown("Ajusta los pesos para crear tu score personalizado:")
    st.sidebar.markdown("💡 *Los pesos se normalizan automáticamente para sumar 100%*")
    
    # Weight sliders for each metric (raw values, will be normalized)
    raw_weights = {}
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.markdown("**Retornos**")
        raw_weights['YTD Return (%)'] = st.slider("YTD Return", 0, 100, 20, 5)
        raw_weights['MTD Return (%)'] = st.slider("MTD Return", 0, 100, 10, 5)
        raw_weights['1Y Return (%)'] = st.slider("1Y Return", 0, 100, 25, 5)
        raw_weights['2024 Return (%)'] = st.slider("2024 Return", 0, 100, 15, 5)
        raw_weights['2023 Return (%)'] = st.slider("2023 Return", 0, 100, 10, 5)
    
    with col2:
        st.markdown("**Riesgo**")
        raw_weights['2022 Return (%)'] = st.slider("2022 Return", 0, 100, 5, 5)
        raw_weights['Monthly Return (%)'] = st.slider("Monthly Return", 0, 100, 5, 5)
        raw_weights['Max Drawdown (%)'] = st.slider("Max Drawdown", 0, 100, 5, 5)
        raw_weights['Volatility (%)'] = st.slider("Volatility", 0, 100, 5, 5)
        raw_weights['VaR 5% (%)'] = st.slider("VaR 5%", 0, 100, 5, 5)
        raw_weights['CVaR 5% (%)'] = st.slider("CVaR 5%", 0, 100, 5, 5)
    
    # Normalize weights to sum to 100%
    total_raw_weight = sum(raw_weights.values())
    
    # Show current raw weight sum with dynamic color
    if total_raw_weight == 100:
        st.sidebar.success(f"🎯 Suma actual: {total_raw_weight}% (Perfecto!)")
    elif total_raw_weight > 100:
        st.sidebar.warning(f"⚠️ Suma actual: {total_raw_weight}% (Se normalizará a 100%)")
    elif total_raw_weight > 0:
        st.sidebar.info(f"📊 Suma actual: {total_raw_weight}% (Se normalizará a 100%)")
    else:
        st.sidebar.error(f"❌ Suma actual: {total_raw_weight}% (Todos los pesos en 0)")
    
    if total_raw_weight > 0:
        weights = {metric: (weight / total_raw_weight) * 100 for metric, weight in raw_weights.items()}
    else:
        # If all weights are 0, distribute equally
        weights = {metric: 100/len(raw_weights) for metric in raw_weights.keys()}
    
    # Show normalization info
    st.sidebar.markdown("**Pesos finales (normalizados):**")
    
    # Show top 5 weights for reference with better formatting
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (metric, weight) in enumerate(sorted_weights):
        metric_short = metric.replace(' (%)', '').replace('Return', 'Ret').replace('Drawdown', 'DD')
        
        # Add emoji for top metrics
        if i == 0:
            emoji = "🥇"
        elif i == 1:
            emoji = "🥈"
        elif i == 2:
            emoji = "🥉"
        else:
            emoji = "📊"
            
        # Color code based on weight
        if weight >= 20:
            st.sidebar.markdown(f"{emoji} **{metric_short}**: **{weight:.1f}%**")
        elif weight >= 10:
            st.sidebar.write(f"{emoji} {metric_short}: {weight:.1f}%")
        else:
            st.sidebar.write(f"{emoji} {metric_short}: {weight:.1f}%", help="Peso menor")
    
    # Z-score explanation
    with st.sidebar.expander("ℹ️ Sobre el Z-Score"):
        st.write("""
        **Z-Score (σ)**: Mide cuántas desviaciones estándar está un fondo por encima o debajo del promedio.
        
        • **+2.0σ**: Excelente (top 2.5%)
        • **+1.0σ**: Muy bueno (top 16%)
        • **0.0σ**: Promedio
        • **-1.0σ**: Bajo promedio (bottom 16%)
        • **-2.0σ**: Muy bajo (bottom 2.5%)
        
        Esto permite comparar métricas de diferentes escalas de forma justa.
        """)
    
    # Date range selector with shortcuts
    st.sidebar.markdown("### 📅 Chart Date Range")
    
    # Get date range from data
    funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
    min_date = funds_data['Dates'].min().date()
    max_date = funds_data['Dates'].max().date()
    
    # Quick date shortcuts
    st.sidebar.markdown("**Quick Shortcuts:**")
    shortcut_cols = st.sidebar.columns(2)
    
    with shortcut_cols[0]:
        if st.button("YTD", use_container_width=True):
            st.session_state.chart_start_date = pd.to_datetime(f'{max_date.year}-01-01').date()
            st.session_state.chart_end_date = max_date
        if st.button("1 Year", use_container_width=True):
            st.session_state.chart_start_date = max_date - timedelta(days=365)
            st.session_state.chart_end_date = max_date
    
    with shortcut_cols[1]:
        if st.button("2 Years", use_container_width=True):
            st.session_state.chart_start_date = max_date - timedelta(days=730)
            st.session_state.chart_end_date = max_date
        if st.button("1 Month", use_container_width=True):
            st.session_state.chart_start_date = max_date - timedelta(days=30)
            st.session_state.chart_end_date = max_date
    
    # Initialize session state for dates if not exists
    if 'chart_start_date' not in st.session_state:
        st.session_state.chart_start_date = max_date - timedelta(days=365*2)
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
    
    # Update session state
    st.session_state.chart_start_date = chart_start_date
    st.session_state.chart_end_date = chart_end_date
    
    # MAIN CONTENT
    if not filtered_funds:
        st.warning("Ningún fondo coincide con los criterios de filtro seleccionados. Ajusta los filtros para ver resultados.")
        return
    
    # Show performance table for all filtered funds
    st.markdown("## 📋 Performance de Fondos Filtrados")
    st.markdown(f"**{len(filtered_funds)} fondos coinciden con los criterios de filtro**")
    
    with st.spinner("Calculando métricas de performance..."):
        performance_data = []
        
        for fund in filtered_funds:
            metrics = calculate_performance_metrics(funds_data, fund)
            
            if metrics:
                fund_display_name = fund_info.get(fund, {}).get('name', fund)
                region = fund_info.get(fund, {}).get('region', 'Unknown')
                asset_class = fund_info.get(fund, {}).get('asset_class', 'Unknown')
                subclass = fund_info.get(fund, {}).get('subclass', 'Unknown')
                sector = fund_info.get(fund, {}).get('sector', 'Unknown')
                
                row = {
                    'Fund Name': fund_display_name,
                    'Ticker': fund,
                    'Region': region,
                    'Asset Class': asset_class,
                    'Subclass': subclass,
                    'Sector': sector,
                    **metrics
                }
                performance_data.append(row)
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            
            # Calculate custom score and sort by it
            df_scored = calculate_custom_score(df_performance, weights)
            
            # Format percentage columns for display
            display_df = df_scored.copy()
            percentage_cols = ['YTD Return (%)', 'MTD Return (%)', 'Monthly Return (%)', '1Y Return (%)',
                             '2024 Return (%)', '2023 Return (%)', '2022 Return (%)', 'Max Drawdown (%)', 
                             'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
            
            for col in percentage_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            
            # Format custom score (Z-score format)
            if 'Custom Score' in display_df.columns:
                display_df['Custom Score'] = display_df['Custom Score'].apply(lambda x: f"{x:.2f}σ" if pd.notnull(x) else "N/A")
            
            # Reorder columns to put Custom Score first after basic info
            base_cols = ['Fund Name', 'Ticker', 'Region', 'Asset Class', 'Subclass', 'Sector']
            if 'Custom Score' in display_df.columns:
                base_cols.append('Custom Score')
            
            metric_cols = [col for col in display_df.columns if col not in base_cols]
            final_cols = base_cols + metric_cols
            display_df = display_df[final_cols]
            
            # Display the performance table with ranking
            st.markdown("**📊 Fondos ordenados por score personalizado**")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Create selection interface using fund names from scored dataframe (for downloads and charts)
            fund_options = {}
            for _, row in df_scored.iterrows():
                display_name = row['Fund Name']
                if len(display_name) > 80:
                    display_name = display_name[:77] + "..."
                fund_options[f"{display_name} ({row['Ticker']})"] = row['Ticker']
            
            # Download buttons section
            st.markdown("### 📥 Descargar Reportes")
            
            download_cols = st.columns(3)
            
            with download_cols[0]:
                # CSV Download
                csv_data = create_csv_report(display_df)
                if csv_data:
                    st.download_button(
                        label="📄 Descargar CSV",
                        data=csv_data,
                        file_name=f"analisis_fondos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with download_cols[1]:
                # Excel Download with charts - include all available fund tickers for comprehensive data
                all_tickers = list(fund_options.values())[:20]  # Limit to 20 funds to avoid Excel size issues
                
                # Calculate frontier data and charts for export if we have enough funds
                frontier_df_export = None
                frontier_chart_export = None
                cumulative_chart_export = None
                
                if len(filtered_funds) >= 2:
                    try:
                        fund_names_dict = {}
                        for _, row in df_scored.iterrows():
                            fund_names_dict[row['Ticker']] = row['Fund Name']
                        
                        frontier_result = create_efficient_frontier_chart(funds_data, filtered_funds, fund_names_dict, df_scored, None)
                        if frontier_result and len(frontier_result) == 2:
                            frontier_chart_export, frontier_df_export = frontier_result
                    except:
                        frontier_df_export = None
                        frontier_chart_export = None
                
                # Generate cumulative returns chart for top 5 funds
                if len(fund_options) > 0:
                    try:
                        top_5_tickers = list(fund_options.values())[:5]
                        chart_end_date = pd.to_datetime(funds_data['Dates'].max())
                        chart_start_date = chart_end_date - timedelta(days=365)  # Last year
                        
                        cumulative_chart_export = create_cumulative_returns_chart(
                            funds_data, 
                            top_5_tickers, 
                            chart_start_date, 
                            chart_end_date
                        )
                    except:
                        cumulative_chart_export = None
                
                excel_data = create_excel_report(
                    df_scored, 
                    cumulative_chart_export, 
                    frontier_chart_export, 
                    weights, 
                    len(filtered_funds), 
                    funds_data, 
                    all_tickers, 
                    frontier_df_export
                )
                if excel_data:
                    st.download_button(
                        label="📊 Descargar Excel",
                        data=excel_data,
                        file_name=f"analisis_fondos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with download_cols[2]:
                # PDF Download with charts
                frontier_df_export = None
                frontier_chart_export = None
                cumulative_chart_export = None
                
                # Generate charts for export
                if len(filtered_funds) >= 2:
                    try:
                        fund_names_dict = {}
                        for _, row in df_scored.iterrows():
                            fund_names_dict[row['Ticker']] = row['Fund Name']
                        
                        frontier_result = create_efficient_frontier_chart(funds_data, filtered_funds, fund_names_dict, df_scored, None)
                        if frontier_result and len(frontier_result) == 2:
                            frontier_chart_export, frontier_df_export = frontier_result
                    except:
                        frontier_df_export = None
                        frontier_chart_export = None
                
                # Generate cumulative returns chart for top 5 funds
                if len(fund_options) > 0:
                    try:
                        top_5_tickers = list(fund_options.values())[:5]
                        chart_end_date = pd.to_datetime(funds_data['Dates'].max())
                        chart_start_date = chart_end_date - timedelta(days=365)  # Last year
                        
                        cumulative_chart_export = create_cumulative_returns_chart(
                            funds_data, 
                            top_5_tickers, 
                            chart_start_date, 
                            chart_end_date
                        )
                    except:
                        cumulative_chart_export = None
                
                pdf_data = create_pdf_report(
                    df_scored, 
                    cumulative_chart_export, 
                    frontier_chart_export, 
                    weights, 
                    len(filtered_funds), 
                    frontier_df_export
                )
                if pdf_data:
                    st.download_button(
                        label="📑 Descargar PDF",
                        data=pdf_data,
                        file_name=f"analisis_fondos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
            # Fund selection for chart
            st.markdown("## 📊 Análisis de Gráficos")
            selected_fund_names = st.multiselect(
                "Seleccionar fondos para análisis gráfico:",
                options=list(fund_options.keys()),
                default=list(fund_options.keys())[:5] if len(fund_options) >= 5 else list(fund_options.keys())
            )
            
            # Get selected fund tickers
            selected_funds = [fund_options[name] for name in selected_fund_names]
            
            if selected_funds:
                st.markdown(f"**Mostrando {len(selected_funds)} fondos seleccionados desde {chart_start_date} hasta {chart_end_date}**")
                
                # Create and display chart with custom date range
                chart = create_cumulative_returns_chart(
                    funds_data, 
                    selected_funds, 
                    pd.to_datetime(chart_start_date), 
                    pd.to_datetime(chart_end_date)
                )
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Show selected funds summary (from scored dataframe)
                selected_performance = df_scored[df_scored['Ticker'].isin(selected_funds)]
                st.markdown("### 📈 Resumen de Fondos Seleccionados")
                
                # Format the summary table for display
                summary_display = selected_performance.copy()
                for col in percentage_cols:
                    if col in summary_display.columns:
                        summary_display[col] = summary_display[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                
                if 'Custom Score' in summary_display.columns:
                    summary_display['Custom Score'] = summary_display['Custom Score'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
                
                st.dataframe(summary_display[final_cols], use_container_width=True, hide_index=True)
            else:
                st.info("👆 Selecciona fondos de la lista para ver el gráfico de retornos acumulados.")
            
            # Análisis de Frontera Eficiente
            st.markdown("## 🎯 Análisis de Frontera Eficiente")
            
            if len(filtered_funds) >= 2:  # Need at least 2 funds for frontier
                # Create fund names dictionary for the chart
                fund_names_dict = {}
                for _, row in df_scored.iterrows():
                    fund_names_dict[row['Ticker']] = row['Fund Name']
                
                # Create efficient frontier chart
                with st.spinner("Calculando frontera eficiente..."):
                    frontier_result = create_efficient_frontier_chart(
                        funds_data, 
                        filtered_funds, 
                        fund_names_dict, 
                        df_scored,
                        None  # No debug container for main calculation
                    )
                
                if frontier_result and len(frontier_result) == 2 and frontier_result[0] is not None:
                    frontier_chart, frontier_df = frontier_result
                    st.plotly_chart(frontier_chart, use_container_width=True)
                    
                    # Display efficient frontier data table
                    if frontier_df is not None and not frontier_df.empty:
                        with st.expander("📊 Datos de Frontera Eficiente"):
                            # Format the dataframe for display
                            display_frontier = frontier_df.copy()
                            
                            # Round numerical columns
                            numerical_cols = display_frontier.select_dtypes(include=[np.number]).columns
                            display_frontier[numerical_cols] = display_frontier[numerical_cols].round(4)
                            
                            # Format percentage columns
                            for col in ['Retorno esperado', 'Volatilidad']:
                                if col in display_frontier.columns:
                                    display_frontier[col] = display_frontier[col].apply(lambda x: f"{x:.2%}")
                            
                            if 'Sharpe Ratio' in display_frontier.columns:
                                display_frontier['Sharpe Ratio'] = display_frontier['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(display_frontier, use_container_width=True, hide_index=True)
                            
                            # Download button for efficient frontier data
                            frontier_csv = frontier_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar Datos CSV",
                                data=frontier_csv,
                                file_name=f"frontera_eficiente_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                else:
                    st.warning("""
                    **No se pudo calcular la frontera eficiente**
                    
                    Posibles causas:
                    • **Datos insuficientes**: Se necesitan al menos 50 observaciones de precios
                    • **Alta correlación**: Los fondos seleccionados están muy correlacionados
                    • **Problemas numéricos**: Matriz de covarianza singular o mal condicionada
                    
                    **Sugerencias:**
                    • Selecciona fondos de diferentes regiones o clases de activos
                    • Verifica que los fondos tengan suficiente historial de datos
                    • Reduce el número de fondos seleccionados (máximo 10)
                    """)
                    
                    # Show diagnostic information
                    with st.expander("🔍 Información de Diagnóstico"):
                        debug_container = st.container()
                        
                        try:
                            # Basic statistics about selected funds
                            available_tickers = [ticker for ticker in selected_funds if ticker in funds_data.columns]
                            if available_tickers:
                                funds_subset = funds_data[['Dates'] + available_tickers].copy()
                                funds_subset['Dates'] = pd.to_datetime(funds_subset['Dates'])
                                funds_subset = funds_subset.dropna()
                                
                                st.write(f"**Fondos disponibles:** {len(available_tickers)}")
                                st.write(f"**Observaciones de datos:** {len(funds_subset)}")
                                
                                if len(funds_subset) > 1:
                                    funds_subset = funds_subset.set_index('Dates')
                                    returns = funds_subset.pct_change().dropna()
                                    
                                    if len(returns) > 1:
                                        corr_matrix = returns.corr()
                                        st.write(f"**Observaciones de retornos:** {len(returns)}")
                                        st.write(f"**Correlación promedio:** {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
                                        st.write(f"**Correlación máxima:** {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")
                                        
                                        # Show correlation matrix
                                        st.write("**Matriz de Correlación:**")
                                        st.dataframe(corr_matrix.round(3))
                                
                                # Try to run the frontier calculation with debug output
                                st.write("---")
                                st.write("**Intentando calcular frontera eficiente:**")
                                
                                fund_names_dict = {}
                                for _, row in selected_performance.iterrows():
                                    fund_names_dict[row['Ticker']] = row['Fund Name']
                                
                                # Run with debug output
                                result = create_efficient_frontier_chart(
                                    funds_data, 
                                    selected_funds, 
                                    fund_names_dict, 
                                    selected_performance,
                                    debug_container
                                )
                                
                        except Exception as e:
                            st.write(f"Error en diagnóstico: {e}")
            else:
                st.info("Selecciona al menos 2 fondos para ver el análisis de frontera eficiente.")
                
        else:
            st.warning("No hay datos de performance disponibles para los fondos filtrados.")

if __name__ == "__main__":
    main()