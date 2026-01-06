#!/usr/bin/env python3
"""
DASHBOARD WITH PORTFOLIO FUNCTIONALITY
Extensión del dashboard original que agrega funcionalidad de portafolios sin modificar el código base
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import io
import sys

from metrics_calculator import calculate_individual_fund_metrics, calculate_portfolio_metrics

# Configuración de página
st.set_page_config(
    page_title="Dashboard de Fondos + Portafolios",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar funciones del dashboard original
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Cargar funciones necesarias del dashboard original
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

# Estilos CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #fafafa;
        margin-bottom: 1rem;
        text-align: center;
    }
    .portfolio-widget {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .fund-row {
        background-color: #262730;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border: 1px solid #404040;
    }
</style>
""", unsafe_allow_html=True)

# Clase para gestión de portafolios
class Portfolio:
    @staticmethod
    def init_session_state():
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
    
    @staticmethod
    def add_fund(ticker, name, category="General"):
        st.session_state.portfolio[ticker] = {
            'name': name,
            'category': category,
            'added': datetime.now()
        }
        Portfolio.rebalance()
        st.success(f"✅ {name} agregado al portafolio")
    
    @staticmethod
    def remove_fund(ticker):
        if ticker in st.session_state.portfolio:
            name = st.session_state.portfolio[ticker]['name']
            del st.session_state.portfolio[ticker]
            if ticker in st.session_state.portfolio_weights:
                del st.session_state.portfolio_weights[ticker]
            Portfolio.rebalance()
            st.success(f"🗑️ {name} removido")
    
    @staticmethod
    def rebalance():
        n = len(st.session_state.portfolio)
        if n > 0:
            weight = 100.0 / n
            for ticker in st.session_state.portfolio:
                st.session_state.portfolio_weights[ticker] = weight
    
    @staticmethod
    def normalize():
        total = sum(st.session_state.portfolio_weights.values())
        if total > 0:
            for ticker in st.session_state.portfolio_weights:
                st.session_state.portfolio_weights[ticker] *= 100 / total
    
    @staticmethod
    def calculate_metrics(funds_df, start_date, end_date):
        if not st.session_state.portfolio:
            return None
        
        try:
            # Obtener tickers y pesos
            selected_funds = list(st.session_state.portfolio.keys())
            weights = st.session_state.portfolio_weights
            
            # Usar el calculador centralizado
            results = calculate_portfolio_metrics(
                funds_df, 
                selected_funds, 
                weights, 
                start_date=pd.to_datetime(start_date), 
                end_date=pd.to_datetime(end_date)
            )
            
            if results is None:
                return None
            
            # Preparar datos para el gráfico
            portfolio_returns = results['portfolio_returns']
            cumulative_returns = (1 + portfolio_returns).cumprod() * 100
            
            metrics = {
                'Retorno Total (%)': results['total_return'],
                'Retorno Anualizado (%)': results['annualized_return'],
                'Volatilidad (%)': results['volatility'],
                'Sharpe Ratio': results['sharpe_ratio'],
                'Max Drawdown (%)': results['max_drawdown'],
                'chart_data': (portfolio_returns.index.tolist(), cumulative_returns.tolist())
            }
            
            # Agregar retornos anuales
            for year in [2025, 2024, 2023]:
                key = f'{year} Return (%)'
                if key in results:
                    metrics[key] = results[key]
            
            return metrics
        except:
            return None
    
    @staticmethod
    def export_excel():
        if not st.session_state.portfolio:
            return None
        
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Portfolio composition
                data = []
                for ticker, info in st.session_state.portfolio.items():
                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                    data.append({
                        'Ticker': ticker,
                        'Nombre': info['name'],
                        'Categoría': info['category'],
                        'Peso (%)': weight
                    })
                
                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name='Portafolio', index=False)
            
            output.seek(0)
            return output.getvalue()
        except:
            return None

def render_portfolio_sidebar():
    """Widget del portafolio en el sidebar"""
    Portfolio.init_session_state()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Mi Portafolio")
    
    n_assets = len(st.session_state.portfolio)
    
    if n_assets > 0:
        st.sidebar.success(f"📊 {n_assets} activo(s)")
        
        total_weight = sum(st.session_state.portfolio_weights.values())
        st.sidebar.metric("Peso Total", f"{total_weight:.1f}%")
        
        # Lista de activos
        for ticker, info in list(st.session_state.portfolio.items())[:3]:
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
        
        if n_assets > 3:
            st.sidebar.caption(f"... y {n_assets-3} más")
        
        # Botones
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔍 Gestionar", use_container_width=True):
                st.session_state.show_portfolio = True
                st.rerun()
        with col2:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.portfolio = {}
                st.session_state.portfolio_weights = {}
                st.rerun()
    else:
        st.sidebar.info("Carrito vacío")

def render_add_button(ticker, name):
    """Botón para agregar fondo al portafolio"""
    Portfolio.init_session_state()
    
    if ticker in st.session_state.portfolio:
        return st.button("✅ En Portafolio", disabled=True, key=f"add_{ticker}")
    else:
        if st.button("➕ Agregar", key=f"add_{ticker}"):
            Portfolio.add_fund(ticker, name)
            st.rerun()

def render_portfolio_manager(funds_data, etf_dict):
    """Panel completo de gestión de portafolios"""
    st.markdown("# 🛒 Gestión de Portafolio")
    
    if st.button("← Volver al Análisis"):
        st.session_state.show_portfolio = False
        st.rerun()
    
    if not st.session_state.portfolio:
        st.info("Tu portafolio está vacío.")
        return
    
    # Editor de pesos
    st.markdown("## ⚖️ Configuración")
    
    with st.form("portfolio_form"):
        new_weights = {}
        
        for ticker, info in st.session_state.portfolio.items():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{info['name']}** ({ticker})")
            
            with col2:
                current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                new_weights[ticker] = st.number_input(
                    "Peso %",
                    0.0, 100.0, current_weight, 0.1,
                    key=f"weight_{ticker}",
                    label_visibility="collapsed"
                )
            
            with col3:
                if st.form_submit_button("🗑️", key=f"remove_{ticker}"):
                    Portfolio.remove_fund(ticker)
                    st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("💾 Actualizar Pesos", use_container_width=True):
                st.session_state.portfolio_weights.update(new_weights)
                st.success("✅ Pesos actualizados")
                st.rerun()
        
        with col2:
            if st.form_submit_button("⚖️ Normalizar 100%", use_container_width=True):
                Portfolio.normalize()
                st.success("✅ Pesos normalizados")
                st.rerun()
    
    # Verificar suma
    total = sum(st.session_state.portfolio_weights.values())
    if abs(total - 100) > 0.1:
        st.warning(f"⚠️ Los pesos suman {total:.1f}%. Considera normalizar.")
    
    # Gráfico de asignación
    st.markdown("## 📊 Asignación")
    
    tickers = list(st.session_state.portfolio.keys())
    names = [st.session_state.portfolio[t]['name'] for t in tickers]
    weights = [st.session_state.portfolio_weights.get(t, 0) for t in tickers]
    
    fig = go.Figure(data=[go.Pie(
        labels=[f"{name}<br>({ticker})" for name, ticker in zip(names, tickers)],
        values=weights,
        hole=0.4
    )])
    
    fig.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fafafa')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de performance
    st.markdown("## 📈 Análisis de Performance")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        start_date = st.date_input("Fecha Inicio", datetime.now().date() - timedelta(days=365))
    with col2:
        end_date = st.date_input("Fecha Fin", datetime.now().date())
    with col3:
        st.write("")
        if st.button("🔄 Calcular", use_container_width=True):
            with st.spinner("Calculando..."):
                metrics = Portfolio.calculate_metrics(funds_data, pd.to_datetime(start_date), pd.to_datetime(end_date))
                
                if metrics:
                    # Mostrar métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Retorno Total", f"{metrics['Retorno Total (%)']:.2f}%")
                    with col2:
                        st.metric("Volatilidad", f"{metrics['Volatilidad (%)']:.2f}%")
                    with col3:
                        st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                    with col4:
                        st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
                    
                    # Mostrar retornos anuales
                    st.markdown("### 📅 Retornos Anuales")
                    ann_col1, ann_col2, ann_col3 = st.columns(3)
                    with ann_col1:
                        if '2025 Return (%)' in metrics:
                            st.metric("2025 Return", f"{metrics['2025 Return (%)']:.2f}%")
                    with ann_col2:
                        if '2024 Return (%)' in metrics:
                            st.metric("2024 Return", f"{metrics['2024 Return (%)']:.2f}%")
                    with ann_col3:
                        if '2023 Return (%)' in metrics:
                            st.metric("2023 Return", f"{metrics['2023 Return (%)']:.2f}%")
                    
                    # Gráfico
                    dates, values = metrics['chart_data']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Portafolio'))
                    fig.update_layout(
                        title="Performance del Portafolio (Base 100)",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#fafafa')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Descarga
                    excel_data = Portfolio.export_excel()
                    if excel_data:
                        st.download_button(
                            "📊 Descargar Excel",
                            excel_data,
                            f"portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("No se pudo calcular la performance.")

def main():
    """Función principal"""
    
    # Inicializar estado
    Portfolio.init_session_state()
    if 'show_portfolio' not in st.session_state:
        st.session_state.show_portfolio = False
    
    # Cargar datos
    funds_data, etf_dict = load_data()
    if funds_data is None or etf_dict is None:
        return
    
    # Sidebar del portafolio
    render_portfolio_sidebar()
    
    # Verificar si mostrar gestión de portafolio
    if st.session_state.get('show_portfolio', False):
        render_portfolio_manager(funds_data, etf_dict)
        return
    
    # Dashboard principal
    st.markdown('<h1 class="main-header">📈 Dashboard de Análisis de Fondos + Portafolios</h1>', unsafe_allow_html=True)
    
    # Filtros básicos
    st.sidebar.markdown("## 🔍 Filtros")
    
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Búsqueda
    search = st.sidebar.text_input("🔍 Buscar fondo:")
    if search:
        fund_columns = [col for col in fund_columns if search.lower() in col.lower()]
    
    # Límite de fondos a mostrar
    max_funds = st.sidebar.slider("Máximo fondos a mostrar:", 10, 100, 20)
    fund_columns = fund_columns[:max_funds]
    
    # Tabla principal con métricas y botones
    st.markdown("## 🏆 Fondos Disponibles")
    
    if not fund_columns:
        st.warning("No se encontraron fondos con los criterios de búsqueda.")
        return
    
    # Calcular métricas para los fondos
    with st.spinner("Calculando métricas..."):
        fund_data = []
        
        for ticker in fund_columns:
            metrics = calculate_performance_metrics(funds_data, ticker)
            if metrics:
                # Obtener nombre del fondo
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
                
                fund_data.append({
                    'Ticker': ticker,
                    'Nombre': fund_name,
                    'YTD (%)': metrics.get('YTD Return (%)', 0),
                    '2025 (%)': metrics.get('2025 Return (%)', 0),
                    '2024 (%)': metrics.get('2024 Return (%)', 0),
                    '1Y (%)': metrics.get('1Y Return (%)', 0),
                    'Vol (%)': metrics.get('Volatility (%)', 0),
                    'DD (%)': metrics.get('Max Drawdown (%)', 0)
                })
        
        if not fund_data:
            st.error("No se pudieron calcular métricas.")
            return
        
        df = pd.DataFrame(fund_data)
        df = df.sort_values('YTD (%)', ascending=False)
    
    # Mostrar fondos con botones
    for _, row in df.iterrows():
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Información del fondo
                metrics_parts = [
                    f"YTD: {row['YTD (%)']:.2f}%",
                    f"2025: {row['2025 (%)']:.2f}%" if row['2025 (%)'] != 0 else None,
                    f"2024: {row['2024 (%)']:.2f}%" if row['2024 (%)'] != 0 else None,
                    f"1Y: {row['1Y (%)']:.2f}%",
                    f"Vol: {row['Vol (%)']:.2f}%",
                    f"DD: {row['DD (%)']:.2f}%"
                ]
                metrics_str = " | ".join([p for p in metrics_parts if p is not None])
                
                st.markdown(f"""
                <div class="fund-row">
                    <strong>{row['Nombre']}</strong> ({row['Ticker']})<br>
                    <small>{metrics_str}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                render_add_button(row['Ticker'], row['Nombre'])

if __name__ == "__main__":
    main()