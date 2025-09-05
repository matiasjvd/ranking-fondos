#!/usr/bin/env python3
"""
MAIN DASHBOARD WITH PORTFOLIO BUILDER
Dashboard principal que integra el análisis de fondos existente con el constructor de portafolios
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import importlib.util

# Agregar el directorio actual al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar funciones específicas del dashboard original sin ejecutar el main
def load_funds_dashboard_functions():
    """Cargar funciones del dashboard original sin ejecutar su main"""
    spec = importlib.util.spec_from_file_location("funds_dashboard", "funds_dashboard.py")
    funds_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(funds_module)
    return funds_module

# Cargar módulo de fondos
funds_module = load_funds_dashboard_functions()

# Importar módulo de portafolios
from portfolio_builder import PortfolioBuilder, render_portfolio_cart_widget, render_add_to_portfolio_button, render_portfolio_management_tab

# Configuración de página
st.set_page_config(
    page_title="Dashboard de Análisis de Fondos + Portafolios",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS (reutilizando los del dashboard original)
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
    
    /* Portfolio cart styling */
    .portfolio-cart {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        color: #fafafa;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
    
    /* Better contrast for charts */
    .js-plotly-plot {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Función principal del dashboard"""
    
    # Inicializar estado de sesión
    if 'show_portfolio_tab' not in st.session_state:
        st.session_state.show_portfolio_tab = False
    
    # Cargar datos
    with st.spinner("Cargando datos..."):
        funds_data, etf_dict = funds_module.load_data()
    
    if funds_data is None or etf_dict is None:
        st.error("No se pudieron cargar los datos. Por favor verifica los archivos.")
        return
    
    # Header principal
    st.markdown('<h1 class="main-header">📈 Dashboard de Análisis de Fondos + Constructor de Portafolios</h1>', unsafe_allow_html=True)
    
    # Widget del carrito en el sidebar
    render_portfolio_cart_widget()
    
    # Crear pestañas
    tab1, tab2 = st.tabs(["📊 Análisis de Fondos", "🛒 Constructor de Portafolios"])
    
    with tab1:
        render_funds_analysis_tab(funds_data, etf_dict)
    
    with tab2:
        render_portfolio_management_tab(funds_data, etf_dict)

def render_funds_analysis_tab(funds_data, etf_dict):
    """Renderizar la pestaña de análisis de fondos (funcionalidad original)"""
    
    # SIDEBAR FILTERS (copiado del dashboard original)
    st.sidebar.markdown("## 🔍 Filtros de Análisis")
    
    # Preparar datos para filtros
    funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Crear diccionario de fondos para el selector
    fund_options = {}
    for ticker in fund_columns:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        if not fund_info.empty and 'Fund Name' in fund_info.columns:
            fund_name = fund_info['Fund Name'].iloc[0]
            fund_options[f"{fund_name} ({ticker})"] = ticker
        else:
            fund_options[ticker] = ticker
    
    # Filtros de categoría (si están disponibles)
    if 'Category' in etf_dict.columns:
        categories = ['Todos'] + sorted(etf_dict['Category'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Categoría:", categories)
        
        if selected_category != 'Todos':
            filtered_tickers = etf_dict[etf_dict['Category'] == selected_category]['Ticker'].tolist()
            fund_options = {k: v for k, v in fund_options.items() if v in filtered_tickers}
    
    # Filtro de búsqueda
    search_term = st.sidebar.text_input("🔍 Buscar fondo:", "")
    if search_term:
        fund_options = {k: v for k, v in fund_options.items() if search_term.lower() in k.lower()}
    
    # Configuración de pesos para scoring
    st.sidebar.markdown("## ⚖️ Configuración de Pesos")
    
    weights = {}
    weights['YTD Return (%)'] = st.sidebar.slider("YTD Return", 0, 100, 20)
    weights['1Y Return (%)'] = st.sidebar.slider("1Y Return", 0, 100, 25)
    weights['2024 Return (%)'] = st.sidebar.slider("2024 Return", 0, 100, 15)
    weights['Max Drawdown (%)'] = st.sidebar.slider("Max Drawdown", 0, 100, 15)
    weights['Volatility (%)'] = st.sidebar.slider("Volatility", 0, 100, 10)
    weights['VaR 5% (%)'] = st.sidebar.slider("VaR 5%", 0, 100, 10)
    weights['CVaR 5% (%)'] = st.sidebar.slider("CVaR 5%", 0, 100, 5)
    
    # Configuración de fechas para gráficos
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
        st.warning("Ningún fondo coincide con los criterios de filtro seleccionados.")
        return
    
    # Calcular métricas de performance
    with st.spinner("Calculando métricas de performance..."):
        performance_data = []
        
        for ticker in filtered_funds:
            metrics = funds_module.calculate_performance_metrics(funds_data, ticker)
            if metrics:
                # Obtener nombre del fondo
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
                
                row = {'Ticker': ticker, 'Fund Name': fund_name}
                row.update(metrics)
                performance_data.append(row)
        
        if not performance_data:
            st.error("No se pudieron calcular métricas para ningún fondo.")
            return
        
        df_performance = pd.DataFrame(performance_data)
    
    # Calcular score personalizado
    df_scored = funds_module.calculate_custom_score(df_performance, weights)
    
    # Mostrar tabla de resultados con botones de agregar al portafolio
    st.markdown("## 🏆 Ranking de Fondos")
    st.markdown(f"Mostrando {len(df_scored)} fondos ordenados por score personalizado")
    
    # Preparar tabla para mostrar
    display_df = df_scored.copy()
    
    # Formatear columnas de porcentaje
    percentage_cols = ['YTD Return (%)', 'Monthly Return (%)', '1Y Return (%)', 
                      '2024 Return (%)', '2023 Return (%)', '2022 Return (%)',
                      'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
    
    for col in percentage_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    
    if 'Custom Score' in display_df.columns:
        display_df['Custom Score'] = display_df['Custom Score'].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A")
    
    # Columnas finales para mostrar
    final_cols = ['Ticker', 'Fund Name', 'Custom Score'] + percentage_cols
    final_cols = [col for col in final_cols if col in display_df.columns]
    
    # Mostrar tabla con botones de agregar
    for idx, row in display_df.head(20).iterrows():  # Mostrar top 20
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Crear una fila de métricas
            metrics_text = " | ".join([f"{col}: {row[col]}" for col in final_cols[2:6] if col in row])
            st.markdown(f"**{row['Fund Name']} ({row['Ticker']})** - Score: {row['Custom Score']}")
            st.caption(metrics_text)
        
        with col2:
            render_add_to_portfolio_button(row['Ticker'], row['Fund Name'], etf_dict)
        
        st.markdown("---")
    
    # Gráfico de retornos acumulados
    st.markdown("## 📊 Análisis de Gráficos")
    
    # Selector de fondos para gráfico
    top_funds = list(df_scored.head(10)['Ticker'])
    fund_names_for_chart = {}
    for ticker in top_funds:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
        fund_names_for_chart[f"{fund_name} ({ticker})"] = ticker
    
    selected_fund_names = st.multiselect(
        "Seleccionar fondos para análisis gráfico:",
        options=list(fund_names_for_chart.keys()),
        default=list(fund_names_for_chart.keys())[:5] if len(fund_names_for_chart) >= 5 else list(fund_names_for_chart.keys())
    )
    
    # Get selected fund tickers
    selected_funds = [fund_names_for_chart[name] for name in selected_fund_names]
    
    if selected_funds:
        st.markdown(f"**Mostrando {len(selected_funds)} fondos seleccionados desde {st.session_state.chart_start_date} hasta {st.session_state.chart_end_date}**")
        
        # Create and display chart with custom date range
        chart_key = f"chart_{st.session_state.chart_start_date}_{st.session_state.chart_end_date}_{len(selected_funds)}"
        
        chart = funds_module.create_cumulative_returns_chart(
            funds_data, 
            selected_funds, 
            pd.to_datetime(st.session_state.chart_start_date), 
            pd.to_datetime(st.session_state.chart_end_date)
        )
        if chart:
            st.plotly_chart(chart, use_container_width=True, key=chart_key)
    else:
        st.info("👆 Selecciona fondos de la lista para ver el gráfico de retornos acumulados.")

if __name__ == "__main__":
    main()