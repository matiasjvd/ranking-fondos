#!/usr/bin/env python3
"""
FUNDS DASHBOARD WITH SIMPLE CART
Dashboard original con carrito simple integrado (checkboxes)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os

import cvxpy as cp

# SHARED METRICS CALCULATOR (Sincronizado en ambos dashboards)
from metrics_calculator import calculate_individual_fund_metrics, calculate_portfolio_metrics

# SIMPLE CART INTEGRATION
from simple_cart_fixed import PortfolioManager, integrate_portfolio_manager





# Page configuration
st.set_page_config(
    page_title="Fund Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)



import hashlib, inspect, metrics_calculator

APP_VERSION = "2026-01-06_11-50"

def _fingerprint():
    src = inspect.getsource(metrics_calculator.calculate_individual_fund_metrics)
    return hashlib.md5(src.encode()).hexdigest()[:10]

st.sidebar.caption(f"APP_VERSION: {APP_VERSION}")
st.sidebar.caption(f"metrics_calculator hash: {_fingerprint()}")



# Professional dark theme styling (IGUAL QUE EL ORIGINAL)
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
def load_data(_prices_mtime: float = None, _dict_mtime: float = None):
    """Load funds data and dictionary from CSV files. Cache se invalida con mtime."""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Load price data with low_memory=False to avoid dtype warnings
        funds_path = os.path.join(data_dir, 'funds_prices.csv')
        funds = pd.read_csv(funds_path, low_memory=False)
        
        # Load dictionary data
        dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
        etf_dict = pd.read_csv(dict_path)
        
        # Debug: print data range to verify it's loading correctly
        if 'Dates' in funds.columns:
            funds['Dates'] = pd.to_datetime(funds['Dates'])
            print(f"DEBUG: Datos cargados - Rango: {funds['Dates'].min()} a {funds['Dates'].max()}")
            print(f"DEBUG: Total filas: {len(funds)}")
        
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
    """
    ✅ VERSIÓN SINCRONIZADA - Wrapper que usa metrics_calculator.py
    Garantiza consistencia entre dashboard principal y análisis de portafolio
    """
    return calculate_individual_fund_metrics(funds_df, fund_ticker)

def calculate_custom_score(df, weights):
    """Calculate custom score based on user-defined weights (IGUAL QUE EL ORIGINAL)"""
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

def create_cumulative_returns_chart(funds_df, selected_funds, start_date, end_date, fund_names_dict=None):
    """Create cumulative returns chart for selected funds with proper fund names"""
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
                    
                    # Get display name from fund_names_dict or use ticker as fallback
                    display_name = fund
                    if fund_names_dict:
                        # Find the key that corresponds to this ticker
                        for key, ticker in fund_names_dict.items():
                            if ticker == fund:
                                display_name = key.split(' (')[0]  # Remove ticker from display name
                                break
                    
                    fig.add_trace(go.Scatter(
                        x=fund_data['Dates'],
                        y=cumulative_returns,
                        mode='lines',
                        name=display_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{display_name}</b><br>Date: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
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

def calculate_efficient_frontier_cvxpy(funds_df, fund_tickers, debug_container=None):
    """
    Calculate efficient frontier using CVXPY optimization - EXACT same logic as original dashboard
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
            
        # Filter data for selected funds - USAR TODA LA HISTORIA DISPONIBLE
        available_tickers = [ticker for ticker in fund_tickers if ticker in funds_df.columns]
        if len(available_tickers) < 2:
            debug_log(f"❌ Fondos disponibles insuficientes: {len(available_tickers)} de {len(fund_tickers)}")
            debug_log(f"Fondos solicitados: {fund_tickers[:5]}...")
            debug_log(f"Fondos disponibles: {available_tickers}")
            return None
            
        # Usar TODA la historia disponible para cada fondo (no filtrar por período de análisis)
        funds_data = funds_df[['Dates'] + available_tickers].copy()
        funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
        
        # Información sobre períodos de datos por fondo
        debug_log("📊 Períodos de datos por fondo:")
        for ticker in available_tickers:
            fund_data = funds_data[['Dates', ticker]].dropna()
            if len(fund_data) > 0:
                start_date = fund_data['Dates'].min().strftime('%Y-%m-%d')
                end_date = fund_data['Dates'].max().strftime('%Y-%m-%d')
                debug_log(f"  {ticker}: {start_date} a {end_date} ({len(fund_data)} observaciones)")
        
        # Para la frontera eficiente, usar solo fechas donde TODOS los fondos tienen datos
        funds_data = funds_data.dropna()
        
        if len(funds_data) < 50:
            debug_log(f"❌ Datos insuficientes después de alinear fechas: {len(funds_data)} observaciones (mínimo 50)")
            debug_log(f"Período común: {funds_data['Dates'].min()} a {funds_data['Dates'].max()}")
            return None
        
        debug_log(f"✅ Período común para frontera eficiente: {funds_data['Dates'].min().strftime('%Y-%m-%d')} a {funds_data['Dates'].max().strftime('%Y-%m-%d')}")
        
        # Prepare data - EXACT same as original
        funds_data = funds_data.set_index('Dates')
        returns = funds_data.pct_change().dropna()
        
        # === EXACT same logic as original ===
        mu_ann = returns.mean() * 252
        cov_ann = returns.cov() * 252
        
        n = len(available_tickers)
        debug_log(f"✅ Procesando {n} fondos con {len(returns)} observaciones de retornos")
        
        # Max Sharpe Portfolio - EXACT same as original
        w_sh = cp.Variable(n)
        ret_sh = mu_ann.values @ w_sh
        var_sh = cp.quad_form(w_sh, cov_ann.values)
        cons = [cp.sum(w_sh) == 1, ret_sh >= 1e-4, w_sh >= 0]
        
        # EXACT same as original - single line solve
        debug_log("🔧 Resolviendo optimización Max Sharpe...")
        cp.Problem(cp.Minimize(var_sh), cons).solve()
        
        if w_sh.value is None:
            debug_log("❌ Optimización Max Sharpe falló")
            return None
            
        w_opt = w_sh.value
        ret_opt = mu_ann.values @ w_opt
        std_opt = np.sqrt(w_opt.T @ cov_ann.values @ w_opt)
        debug_log(f"✅ Max Sharpe calculado: Retorno={ret_opt:.4f}, Volatilidad={std_opt:.4f}")
        
        # Generate efficient frontier - EXACT same as original
        targets, data, frontier = np.linspace(mu_ann.min(), mu_ann.max(), 50), [], []
        successful_points = 0
        
        debug_log(f"🔧 Generando frontera eficiente con {len(targets)} puntos...")
        
        for i, t in enumerate(targets):
            w = cp.Variable(n)
            risk = cp.quad_form(w, cov_ann.values)
            c = [cp.sum(w) == 1, mu_ann.values @ w == t, w >= 0]
            try:
                # EXACT same as original - single line solve
                cp.Problem(cp.Minimize(risk), c).solve()
                
                if w.value is not None and risk.value is not None:
                    std = np.sqrt(risk.value)
                    sr = t / std if std > 0 else 0
                    # EXACT same structure as original
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
                # EXACT same as original - simple continue
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

def create_efficient_frontier_chart(funds_df, fund_tickers, fund_names_dict, debug_container=None):
    """Create efficient frontier chart - EXACT same logic as original dashboard"""
    try:
        # Calculate efficient frontier with CVXPY - same as original
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
        
        # Create figure - EXACT same as original
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
        
        # Update layout - dark theme optimized (ESTILO ORIGINAL)
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

def main():
    """Main dashboard function with integrated simple cart"""
    
    # Enlace a Macro Dashboard BVC (botón estático arriba, abre en nueva pestaña)
    st.sidebar.markdown(
        """
        <a href=\"https://dashboard-macro-bvc.streamlit.app/\" target=\"_blank\" rel=\"noopener noreferrer\"
           style=\"display:block; text-align:center; padding:0.6rem 1rem; border-radius:8px; border:1px solid #475569; background-color:#334155; color:#f1f5f9; text-decoration:none; font-weight:600; margin-bottom:0.75rem;\">\n            🌐 Macro Dashboard BVC\n        </a>
        """,
        unsafe_allow_html=True
    )
    
    # INTEGRAR GESTOR DE PORTAFOLIO
    if integrate_portfolio_manager():
        return  # Si se muestra el análisis del portafolio, no mostrar el dashboard principal
    
    # Load data (pasando mtime para invalidar cache cuando cambien los CSV)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
    prices_mtime = os.path.getmtime(funds_path) if os.path.exists(funds_path) else None
    dict_mtime = os.path.getmtime(dict_path) if os.path.exists(dict_path) else None

    with st.spinner("Loading data..."):
        funds_data, etf_dict = load_data(prices_mtime, dict_mtime)
    
    if funds_data is None or etf_dict is None:
        return
    
    # Header
    st.markdown('<h1 class="main-header">Fund Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # SIDEBAR FILTERS
    st.sidebar.markdown("## Analysis Filters")

    # Clear cache button
    if st.sidebar.button("🗑️ Clear Cache"):
        load_data.clear()
        calculate_performance_metrics.clear()
        st.success("Cache cleared. Reloading...")
        st.rerun()

    # Refresh data: rebuild CSVs from Excel in repo and clear cache
    if st.sidebar.button("🔄 Refresh data (rebuild CSVs)"):
        try:
            from convert_data import convert_excel_to_csv
            script_dir = os.path.dirname(os.path.abspath(__file__))
            dict_xlsx = os.path.join(script_dir, 'dict_temp_full_portfolio.xlsx')
            new_prices_xlsx = os.path.join(script_dir, 'Swiss_stocks_10-09-2025.xlsx')
            convert_excel_to_csv(dict_xlsx_path=dict_xlsx, new_prices_xlsx=new_prices_xlsx)
            # Clear cached data and rerun
            load_data.clear()
            calculate_performance_metrics.clear()
            st.success("Data refreshed. Reloading...")
            st.rerun()
        except Exception as e:
            st.error(f"Error refreshing data: {e}")
    
    # Prepare data for filters
    funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
    fund_columns = [col for col in funds_data.columns if col != 'Dates' and not col.startswith('Unnamed')]
    
    # Debug info - show data range
    st.sidebar.markdown("### 📊 Data Info")
    st.sidebar.info(f"**Rango de datos:** {funds_data['Dates'].min().strftime('%Y-%m-%d')} a {funds_data['Dates'].max().strftime('%Y-%m-%d')}")
    st.sidebar.info(f"**Total filas:** {len(funds_data):,}")
    st.sidebar.info(f"**Total fondos:** {len(fund_columns):,}")
    
    # Create fund options dictionary for better display
    fund_options = {}
    for ticker in fund_columns:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        if not fund_info.empty:
            # Try different possible name columns
            fund_name = ticker  # Default to ticker
            # Prioritize "Indice" for more intuitive fund names
            if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                fund_name = fund_info['Indice'].iloc[0]
            elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                fund_name = fund_info['Fund Name'].iloc[0]
            fund_options[f"{fund_name} ({ticker})"] = ticker
        else:
            fund_options[ticker] = ticker
    
    # Initialize filter variables
    selected_asset_class = 'All'
    selected_geography = 'All'
    selected_market = 'All'
    
    # Asset Class filter
    if 'Asset Class' in etf_dict.columns:
        asset_classes = ['All'] + sorted(etf_dict['Asset Class'].dropna().unique().tolist())
        selected_asset_class = st.sidebar.selectbox("🏛️ Asset Class:", asset_classes)
        
        if selected_asset_class != 'All':
            filtered_tickers = etf_dict[etf_dict['Asset Class'] == selected_asset_class]['Ticker'].tolist()
            fund_options = {k: v for k, v in fund_options.items() if v in filtered_tickers}
    
    # Geography / Subclass filter
    if 'Geography / Subclass' in etf_dict.columns:
        # Filter geographies based on selected asset class
        current_tickers = [v for v in fund_options.values()]
        available_geographies = etf_dict[etf_dict['Ticker'].isin(current_tickers)]['Geography / Subclass'].dropna().unique()
        
        geographies = ['All'] + sorted(available_geographies.tolist())
        selected_geography = st.sidebar.selectbox("🌍 Geography / Subclass:", geographies)
        
        if selected_geography != 'All':
            filtered_tickers = etf_dict[etf_dict['Geography / Subclass'] == selected_geography]['Ticker'].tolist()
            fund_options = {k: v for k, v in fund_options.items() if v in filtered_tickers}
    
    # Market filter
    if 'Market' in etf_dict.columns:
        # Filter markets based on previous selections
        current_tickers = [v for v in fund_options.values()]
        available_markets = etf_dict[etf_dict['Ticker'].isin(current_tickers)]['Market'].dropna().unique()
        
        markets = ['All'] + sorted(available_markets.tolist())
        selected_market = st.sidebar.selectbox("🏭 Market:", markets)
        
        if selected_market != 'All':
            filtered_tickers = etf_dict[etf_dict['Market'] == selected_market]['Ticker'].tolist()
            fund_options = {k: v for k, v in fund_options.items() if v in filtered_tickers}
    
    # Search filter
    search_term = st.sidebar.text_input("🔍 Search fund:", "")
    if search_term:
        fund_options = {k: v for k, v in fund_options.items() if search_term.lower() in k.lower()}
    
    # Performance filter
    st.sidebar.markdown("### 📊 Performance Filters")
    min_ytd = st.sidebar.number_input("Min YTD Return (%):", value=-100.0, step=1.0)
    max_volatility = st.sidebar.number_input("Max Volatility (%):", value=100.0, step=1.0)
    
    # Scoring weights configuration (IGUAL QUE EL ORIGINAL)
    st.sidebar.markdown("## ⚖️ Scoring Weights")
    
    weights = {}
    weights['YTD Return (%)'] = st.sidebar.slider("YTD Return", 0, 100, 25)
    weights['2025 Return (%)'] = st.sidebar.slider("2025 Return", 0, 100, 15)
    weights['2024 Return (%)'] = st.sidebar.slider("2024 Return", 0, 100, 15)
    weights['2023 Return (%)'] = st.sidebar.slider("2023 Return", 0, 100, 10)
    weights['Max Drawdown (%)'] = st.sidebar.slider("Max Drawdown", 0, 100, 15)
    weights['Volatility (%)'] = st.sidebar.slider("Volatility", 0, 100, 10)
    weights['VaR 5% (%)'] = st.sidebar.slider("VaR 5%", 0, 100, 10)
    weights['CVaR 5% (%)'] = st.sidebar.slider("CVaR 5%", 0, 100, 5)
    
    # Chart date configuration (IGUAL QUE EL ORIGINAL)
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
                # Get fund name - prioritize "Indice" for more intuitive names
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                fund_name = ticker  # Default
                if not fund_info.empty:
                    if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                        fund_name = fund_info['Indice'].iloc[0]
                    elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                        fund_name = fund_info['Fund Name'].iloc[0]
                
                # Apply performance filters
                if (metrics['YTD Return (%)'] >= min_ytd and 
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
    
    # Prepare display dataframe
    display_df = df_scored.copy()
    
    # Format percentage columns
    percentage_cols = ['YTD Return (%)', 'Monthly Return (%)', '1Y Return (%)',
                      '2025 Return (%)', '2024 Return (%)', '2023 Return (%)',
                      'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
    
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    # Format Sharpe if present
    if 'Sharpe Ratio' in display_df.columns:
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
    
    if 'Custom Score' in display_df.columns:
        display_df['Custom Score'] = display_df['Custom Score'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
    
    # MOSTRAR PRIMERO LA TABLA COMPLETA
    st.markdown("## Fund Analysis Results")
    final_cols = ['Ticker', 'Fund Name', 'Custom Score', 'Sharpe Ratio'] + percentage_cols
    final_cols = [col for col in final_cols if col in display_df.columns]
    st.dataframe(display_df[final_cols], use_container_width=True)
    
    # FUND RANKINGS WITH PORTFOLIO SELECTION
    st.markdown("## Fund Rankings & Portfolio Selection")
    st.markdown(f"Select funds to add to your portfolio. Showing top {min(20, len(df_scored))} funds ordered by custom score.")
    
    # MOSTRAR FONDOS CON CHECKBOXES SIMPLES
    for idx, row in display_df.head(20).iterrows():  # Show top 20
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Fund information
                score_text = f"**Score: {row['Custom Score']}**" if 'Custom Score' in row else ""
                st.markdown(f"**{row['Fund Name']}** ({row['Ticker']}) {score_text}")
                
                # Fund classification from dictionary
                fund_info = etf_dict[etf_dict['Ticker'] == row['Ticker']]
                if not fund_info.empty:
                    classification_parts = []
                    if 'Asset Class' in fund_info.columns and pd.notna(fund_info['Asset Class'].iloc[0]):
                        classification_parts.append(f"{fund_info['Asset Class'].iloc[0]}")
                    if 'Geography / Subclass' in fund_info.columns and pd.notna(fund_info['Geography / Subclass'].iloc[0]):
                        classification_parts.append(f"{fund_info['Geography / Subclass'].iloc[0]}")
                    if 'Market' in fund_info.columns and pd.notna(fund_info['Market'].iloc[0]):
                        classification_parts.append(f"{fund_info['Market'].iloc[0]}")
                    
                    if classification_parts:
                        st.caption(" | ".join(classification_parts))
                
                # Key metrics
                metrics_text = []
                if 'YTD Return (%)' in row:
                    metrics_text.append(f"YTD: {row['YTD Return (%)']}")
                if '2025 Return (%)' in row:
                    metrics_text.append(f"2025: {row['2025 Return (%)']}")
                if '2024 Return (%)' in row:
                    metrics_text.append(f"2024: {row['2024 Return (%)']}")
                if '1Y Return (%)' in row:
                    metrics_text.append(f"1Y: {row['1Y Return (%)']}")
                if 'Volatility (%)' in row:
                    metrics_text.append(f"Vol: {row['Volatility (%)']}")
                if 'Sharpe Ratio' in row:
                    metrics_text.append(f"Sharpe: {row['Sharpe Ratio']}")
                if 'Max Drawdown (%)' in row:
                    metrics_text.append(f"DD: {row['Max Drawdown (%)']}")
                
                if metrics_text:
                    st.caption(" | ".join(metrics_text))
            
            with col2:
                # CHECKBOX SIMPLE PARA SELECCIONAR
                # Get proper fund name from "Indice" column
                fund_info = etf_dict[etf_dict['Ticker'] == row['Ticker']]
                display_name = row['Ticker']  # Default
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
    

    
    # CUMULATIVE RETURNS CHART
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
        
        # Create and display chart with custom date range
        chart_key = f"chart_{st.session_state.chart_start_date}_{st.session_state.chart_end_date}_{len(selected_funds)}"
        
        chart = create_cumulative_returns_chart(
            funds_data, 
            selected_funds, 
            pd.to_datetime(st.session_state.chart_start_date), 
            pd.to_datetime(st.session_state.chart_end_date),
            fund_names_for_chart
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True, key=chart_key)
    else:
        st.info("Select funds from the list above to view cumulative returns chart.")
    
    # EFFICIENT FRONTIER ANALYSIS
    st.markdown("## Efficient Frontier Analysis")
    
    st.info("**Note**: The efficient frontier uses the complete historical data available for each fund to obtain better estimates of correlations and volatilities, not just the selected analysis period.")
    
    # Opción para usar todos los fondos o solo los seleccionados
    frontier_option = st.radio(
        "Select funds for efficient frontier:",
        ["Top 10 funds (by score)", "Selected funds for chart", "All available funds"],
        index=0
    )
    
    # Determinar qué fondos usar
    if frontier_option == "Top 10 funds (by score)":
        frontier_funds = list(df_scored.head(10)['Ticker'])
        st.info(f"Using top 10 funds by custom score")
    elif frontier_option == "Selected funds for chart":
        frontier_funds = selected_funds
        if len(frontier_funds) < 2:
            st.warning("Select at least 2 funds for the chart to calculate the efficient frontier.")
            frontier_funds = []
    else:  # All funds
        frontier_funds = list(df_scored.head(20)['Ticker'])  # Limit to top 20 for efficiency
        st.info(f"Using top 20 available funds for computational efficiency")
    
    if len(frontier_funds) >= 2:
        if st.button("Calculate Efficient Frontier", use_container_width=True):
            with st.spinner("Calculating efficient frontier..."):
                # Crear diccionario de nombres para el gráfico
                fund_names_dict = {}
                for ticker in frontier_funds:
                    fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                    if not fund_info.empty:
                        # Prioritize "Indice" for more intuitive fund names
                        if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                            fund_names_dict[ticker] = fund_info['Indice'].iloc[0]
                        elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                            fund_names_dict[ticker] = fund_info['Fund Name'].iloc[0]
                        else:
                            fund_names_dict[ticker] = ticker
                    if ticker not in fund_names_dict:
                        fund_names_dict[ticker] = ticker
                
                # Debug container
                debug_container = st.empty()
                
                # Calcular frontera eficiente usando TODA la historia disponible (no filtrada por fechas)
                # Esto es importante porque cada fondo tiene diferente historia y queremos usar toda la información
                fig_frontier, frontier_df = create_efficient_frontier_chart(
                    funds_data,  # Usar toda la historia disponible, no filtrada por período de análisis
                    frontier_funds, 
                    fund_names_dict, 
                    debug_container
                )
                
                if fig_frontier:
                    st.plotly_chart(fig_frontier, use_container_width=True)
                    
                    # Mostrar portafolios óptimos
                    if frontier_df is not None and not frontier_df.empty:
                        st.markdown("### 🎯 Portafolios Óptimos")
                        
                        # Encontrar portafolio con mayor Sharpe ratio
                        max_sharpe_idx = frontier_df['Sharpe Ratio'].idxmax()
                        min_vol_idx = frontier_df['Volatilidad'].idxmin()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**🏆 Portafolio Max Sharpe Ratio:**")
                            max_sharpe_portfolio = frontier_df.loc[max_sharpe_idx]
                            st.metric("Retorno Esperado", f"{max_sharpe_portfolio['Retorno esperado']:.2%}")
                            st.metric("Volatilidad", f"{max_sharpe_portfolio['Volatilidad']:.2%}")
                            st.metric("Sharpe Ratio", f"{max_sharpe_portfolio['Sharpe Ratio']:.3f}")
                        
                        with col2:
                            st.markdown("**🛡️ Portafolio Mínima Volatilidad:**")
                            min_vol_portfolio = frontier_df.loc[min_vol_idx]
                            st.metric("Retorno Esperado", f"{min_vol_portfolio['Retorno esperado']:.2%}")
                            st.metric("Volatilidad", f"{min_vol_portfolio['Volatilidad']:.2%}")
                            st.metric("Sharpe Ratio", f"{min_vol_portfolio['Sharpe Ratio']:.3f}")
                        
                        # Mostrar composición del portafolio óptimo
                        st.markdown("**Composición del Portafolio Max Sharpe:**")
                        weights_data = []
                        for ticker in frontier_funds:
                            if ticker in max_sharpe_portfolio and max_sharpe_portfolio[ticker] > 0.001:  # Solo pesos significativos
                                weight = max_sharpe_portfolio[ticker] * 100
                                fund_name = fund_names_dict.get(ticker, ticker)
                                weights_data.append({
                                    'Fondo': fund_name,
                                    'Ticker': ticker,
                                    'Peso (%)': weight
                                })
                        
                        if weights_data:
                            weights_df = pd.DataFrame(weights_data)
                            weights_df = weights_df.sort_values('Peso (%)', ascending=False)
                            st.dataframe(weights_df, use_container_width=True)
                    
                    st.success("💡 La frontera eficiente muestra las combinaciones óptimas de riesgo-retorno. Los puntos en la frontera representan portafolios que maximizan el retorno para un nivel dado de riesgo.")
                
                else:
                    st.error("No se pudo calcular la frontera eficiente. Verifica que los fondos seleccionados tengan datos históricos suficientes.")
                
                # Limpiar debug container
                debug_container.empty()
    
    else:
        st.info("💡 Selecciona al menos 2 fondos para calcular la frontera eficiente.")

if __name__ == "__main__":
    main()