#!/usr/bin/env python3
"""
FUNDS DASHBOARD WITH INTEGRATED PORTFOLIO CART
Dashboard original con carrito de portafolios integrado sin modificar el código base
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Importar el módulo del carrito
from portfolio_cart import PortfolioCart, integrate_portfolio_cart

# Importar y ejecutar el dashboard original con modificaciones mínimas
import importlib.util

def load_and_modify_original_dashboard():
    """Cargar el dashboard original y modificarlo para incluir el carrito"""
    
    # Leer el contenido del dashboard original
    with open('funds_dashboard.py', 'r', encoding='utf-8') as f:
        original_code = f.read()
    
    # Modificaciones mínimas para integrar el carrito
    modified_code = original_code
    
    # 1. Agregar import del carrito después de los imports existentes
    import_insertion = """
# PORTFOLIO CART INTEGRATION
from portfolio_cart import PortfolioCart, integrate_portfolio_cart
"""
    
    # Insertar después de los imports
    import_pos = modified_code.find('warnings.filterwarnings(\'ignore\')')
    if import_pos != -1:
        end_pos = modified_code.find('\n', import_pos) + 1
        modified_code = modified_code[:end_pos] + import_insertion + modified_code[end_pos:]
    
    # 2. Modificar la función main para incluir el carrito
    main_function_start = modified_code.find('def main():')
    if main_function_start != -1:
        # Encontrar el inicio del contenido de main
        main_content_start = modified_code.find('"""', main_function_start)
        main_content_start = modified_code.find('"""', main_content_start + 3) + 3
        
        # Insertar la integración del carrito
        cart_integration = """
    
    # INTEGRAR CARRITO DE PORTAFOLIOS
    if integrate_portfolio_cart():
        return  # Si se muestra la pestaña de portafolio, no mostrar el dashboard principal
    
"""
        modified_code = modified_code[:main_content_start] + cart_integration + modified_code[main_content_start:]
    
    # 3. Modificar la tabla de resultados para incluir botones de agregar al carrito
    # Buscar donde se muestra la tabla de resultados
    table_section = 'st.dataframe('
    table_pos = modified_code.find(table_section)
    
    if table_pos != -1:
        # Reemplazar la tabla con una versión que incluya botones
        table_replacement = """
        # TABLA CON BOTONES DE CARRITO
        st.markdown("**Haz clic en 'Agregar' para añadir fondos a tu carrito de portafolio**")
        
        # Mostrar tabla con botones
        for idx, row in display_df.head(20).iterrows():
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    # Crear string con métricas principales
                    metrics_str = f"Score: {row.get('Custom Score', 'N/A')} | "
                    metrics_str += f"YTD: {row.get('YTD Return (%)', 'N/A')} | "
                    metrics_str += f"1Y: {row.get('1Y Return (%)', 'N/A')} | "
                    metrics_str += f"Vol: {row.get('Volatility (%)', 'N/A')}"
                    
                    st.markdown(f"**{row.get('Fund Name', row.get('Ticker', 'N/A'))}** ({row.get('Ticker', 'N/A')})")
                    st.caption(metrics_str)
                
                with col2:
                    # Botón para agregar al carrito
                    PortfolioCart.render_add_button(
                        row.get('Ticker', ''), 
                        row.get('Fund Name', row.get('Ticker', '')), 
                        key_suffix=f"table_{idx}"
                    )
                
                st.markdown("---")
        
        # También mostrar la tabla completa para referencia
        with st.expander("📊 Ver tabla completa de datos"):
            st.dataframe("""
    
    # Insertar antes de st.dataframe
    modified_code = modified_code[:table_pos] + table_replacement + modified_code[table_pos:]
    
    return modified_code

def execute_modified_dashboard():
    """Ejecutar el dashboard modificado"""
    try:
        modified_code = load_and_modify_original_dashboard()
        
        # Crear un namespace para ejecutar el código
        namespace = {
            '__name__': '__main__',
            'st': st,
            'pd': pd,
            'go': go,
            'px': px,
            'np': np,
            'datetime': datetime,
            'timedelta': timedelta,
            'os': os,
            'PortfolioCart': PortfolioCart,
            'integrate_portfolio_cart': integrate_portfolio_cart
        }
        
        # Importar módulos adicionales que pueda necesitar el dashboard original
        try:
            import cvxpy as cp
            namespace['cp'] = cp
        except ImportError:
            pass
        
        try:
            import base64
            import io
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            import warnings
            
            namespace.update({
                'base64': base64,
                'io': io,
                'letter': letter,
                'A4': A4,
                'SimpleDocTemplate': SimpleDocTemplate,
                'Paragraph': Paragraph,
                'Spacer': Spacer,
                'Table': Table,
                'TableStyle': TableStyle,
                'Image': Image,
                'getSampleStyleSheet': getSampleStyleSheet,
                'ParagraphStyle': ParagraphStyle,
                'inch': inch,
                'colors': colors,
                'warnings': warnings
            })
        except ImportError:
            pass
        
        # Ejecutar el código modificado
        exec(modified_code, namespace)
        
    except Exception as e:
        st.error(f"Error ejecutando dashboard modificado: {e}")
        st.info("Ejecutando versión simplificada...")
        execute_simple_dashboard()

def execute_simple_dashboard():
    """Versión simplificada del dashboard con carrito"""
    
    # Configuración de página
    st.set_page_config(
        page_title="Dashboard de Fondos + Carrito",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Integrar carrito
    if integrate_portfolio_cart():
        return  # Si se muestra la pestaña de portafolio, salir
    
    # Cargar datos (función simplificada)
    @st.cache_data
    def load_data():
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            
            funds_path = os.path.join(data_dir, 'funds_prices.csv')
            funds = pd.read_csv(funds_path)
            
            dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
            etf_dict = pd.read_csv(dict_path)
            
            return funds, etf_dict
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return None, None
    
    # Función simplificada para calcular métricas
    def calculate_basic_metrics(funds_df, ticker):
        try:
            if ticker not in funds_df.columns:
                return None
            
            prices = funds_df[['Dates', ticker]].dropna()
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
            
            # 1 Year Return
            year_1_start = current_date - timedelta(days=365)
            year_1_data = prices[prices['Dates'] >= year_1_start]
            return_1y = ((year_1_data[ticker].iloc[-1] / year_1_data[ticker].iloc[0]) - 1) * 100 if len(year_1_data) > 1 else 0
            
            # Volatility
            volatility = prices['Returns'].std() * np.sqrt(252) * 100
            
            return {
                'YTD Return (%)': ytd_return,
                '1Y Return (%)': return_1y,
                'Volatility (%)': volatility
            }
        except:
            return None
    
    # Header
    st.markdown("# 📈 Dashboard de Análisis de Fondos + Carrito de Portafolios")
    st.markdown("**Versión integrada con funcionalidad de carrito de portafolios**")
    
    # Cargar datos
    funds_data, etf_dict = load_data()
    if funds_data is None or etf_dict is None:
        return
    
    # Sidebar básico
    st.sidebar.markdown("## 🔍 Filtros Básicos")
    
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Búsqueda
    search_term = st.sidebar.text_input("🔍 Buscar fondo:")
    if search_term:
        fund_columns = [col for col in fund_columns if search_term.lower() in col.lower()]
    
    # Límite
    max_display = st.sidebar.slider("Máximo fondos a mostrar:", 10, 50, 20)
    fund_columns = fund_columns[:max_display]
    
    # Contenido principal
    st.markdown("## 🏆 Fondos Disponibles")
    st.markdown("**Haz clic en '🛒 Agregar' para añadir fondos a tu carrito de portafolio**")
    
    if not fund_columns:
        st.warning("No se encontraron fondos con los criterios de búsqueda.")
        return
    
    # Calcular métricas y mostrar fondos
    with st.spinner("Cargando fondos..."):
        for i, ticker in enumerate(fund_columns):
            # Obtener información del fondo
            fund_info = etf_dict[etf_dict['Ticker'] == ticker]
            fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
            
            # Calcular métricas básicas
            metrics = calculate_basic_metrics(funds_data, ticker)
            
            # Mostrar fondo con botón
            with st.container():
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.markdown(f"**{fund_name}** ({ticker})")
                    if metrics:
                        metrics_str = f"YTD: {metrics['YTD Return (%)']:.2f}% | "
                        metrics_str += f"1Y: {metrics['1Y Return (%)']:.2f}% | "
                        metrics_str += f"Vol: {metrics['Volatility (%)']:.2f}%"
                        st.caption(metrics_str)
                    else:
                        st.caption("Métricas no disponibles")
                
                with col2:
                    PortfolioCart.render_add_button(ticker, fund_name, key_suffix=f"main_{i}")
                
                st.markdown("---")
    
    # Información adicional
    st.markdown("## 💡 Cómo usar el carrito")
    st.info("""
    1. **Agregar fondos**: Haz clic en '🛒 Agregar' junto a los fondos que te interesen
    2. **Ver carrito**: Revisa el widget del carrito en el sidebar
    3. **Gestionar portafolio**: Haz clic en '📊 Ver Portafolio' para gestionar pesos y analizar performance
    4. **Exportar**: Descarga tu portafolio en formato Excel
    """)

if __name__ == "__main__":
    # Intentar ejecutar la versión modificada del dashboard original
    execute_modified_dashboard()