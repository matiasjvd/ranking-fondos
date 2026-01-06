#!/usr/bin/env python3
"""
ENHANCED FUNDS DASHBOARD WITH PORTFOLIO BUILDER
Versión extendida del dashboard original que agrega funcionalidad de portafolios
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os
import io
import base64

# Importar todas las funciones del dashboard original
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Ejecutar el dashboard original pero capturar sus funciones
exec(open('funds_dashboard.py').read())

# Clase para manejar portafolios
class PortfolioManager:
    """Gestor de portafolios integrado"""
    
    @staticmethod
    def initialize_portfolio_state():
        """Inicializar estado del portafolio"""
        if 'portfolio_assets' not in st.session_state:
            st.session_state.portfolio_assets = {}
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'show_portfolio_panel' not in st.session_state:
            st.session_state.show_portfolio_panel = False
    
    @staticmethod
    def add_to_portfolio(ticker, fund_name, category="General"):
        """Agregar activo al portafolio"""
        st.session_state.portfolio_assets[ticker] = {
            'name': fund_name,
            'category': category,
            'date_added': datetime.now()
        }
        PortfolioManager.rebalance_weights()
        st.success(f"✅ {fund_name} agregado al portafolio")
    
    @staticmethod
    def remove_from_portfolio(ticker):
        """Remover activo del portafolio"""
        if ticker in st.session_state.portfolio_assets:
            name = st.session_state.portfolio_assets[ticker]['name']
            del st.session_state.portfolio_assets[ticker]
            if ticker in st.session_state.portfolio_weights:
                del st.session_state.portfolio_weights[ticker]
            PortfolioManager.rebalance_weights()
            st.success(f"🗑️ {name} removido del portafolio")
    
    @staticmethod
    def rebalance_weights():
        """Rebalancear pesos equitativamente"""
        num_assets = len(st.session_state.portfolio_assets)
        if num_assets > 0:
            equal_weight = 100.0 / num_assets
            for ticker in st.session_state.portfolio_assets.keys():
                st.session_state.portfolio_weights[ticker] = equal_weight
    
    @staticmethod
    def normalize_weights():
        """Normalizar pesos para que sumen 100%"""
        total = sum(st.session_state.portfolio_weights.values())
        if total > 0:
            for ticker in st.session_state.portfolio_weights:
                st.session_state.portfolio_weights[ticker] = (
                    st.session_state.portfolio_weights[ticker] / total * 100
                )
    
    @staticmethod
    def calculate_portfolio_performance(funds_df, start_date, end_date):
        """Calcular performance del portafolio"""
        if not st.session_state.portfolio_assets:
            return None, None
        
        try:
            # Filtrar datos por fechas
            funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
            filtered_data = funds_df[
                (funds_df['Dates'] >= start_date) & 
                (funds_df['Dates'] <= end_date)
            ].copy()
            
            if filtered_data.empty:
                return None, None
            
            # Calcular valor del portafolio
            portfolio_values = []
            dates = []
            
            for _, row in filtered_data.iterrows():
                portfolio_value = 0
                total_weight = 0
                
                for ticker in st.session_state.portfolio_assets.keys():
                    if ticker in filtered_data.columns and pd.notna(row[ticker]):
                        weight = st.session_state.portfolio_weights.get(ticker, 0) / 100
                        portfolio_value += row[ticker] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_values.append(portfolio_value / total_weight)
                    dates.append(row['Dates'])
            
            if len(portfolio_values) < 2:
                return None, None
            
            # Crear serie de precios del portafolio
            portfolio_series = pd.Series(portfolio_values, index=dates)
            returns = portfolio_series.pct_change().dropna()
            
            # Calcular métricas
            total_return = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1) * 100
            volatility = returns.std() * np.sqrt(252) * 100
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # VaR y CVaR
            var_5 = np.percentile(returns, 5) * np.sqrt(252) * 100
            cvar_5 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252) * 100
            
            metrics = {
                'Retorno Total (%)': total_return,
                'Volatilidad (%)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown,
                'VaR 5% (%)': var_5,
                'CVaR 5% (%)': cvar_5
            }
            
            # Normalizar valores para gráfico (base 100)
            base_value = portfolio_values[0]
            normalized_values = [(v / base_value) * 100 for v in portfolio_values]
            
            return metrics, (dates, normalized_values)
            
        except Exception as e:
            st.error(f"Error calculando performance: {e}")
            return None, None
    
    @staticmethod
    def create_portfolio_chart(dates, values):
        """Crear gráfico del portafolio"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portafolio',
            line=dict(color='#60a5fa', width=3),
            hovertemplate='<b>Portafolio</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Performance del Portafolio (Base 100)",
            xaxis_title="Fecha",
            yaxis_title="Valor",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#404040', color='#fafafa'),
            yaxis=dict(gridcolor='#404040', color='#fafafa')
        )
        
        return fig
    
    @staticmethod
    def create_allocation_chart():
        """Crear gráfico de asignación"""
        if not st.session_state.portfolio_assets:
            return None
        
        tickers = list(st.session_state.portfolio_assets.keys())
        names = [st.session_state.portfolio_assets[t]['name'] for t in tickers]
        weights = [st.session_state.portfolio_weights.get(t, 0) for t in tickers]
        
        fig = go.Figure(data=[go.Pie(
            labels=[f"{name}<br>({ticker})" for name, ticker in zip(names, tickers)],
            values=weights,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=px.colors.qualitative.Set3)
        )])
        
        fig.update_layout(
            title="Asignación del Portafolio",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa')
        )
        
        return fig
    
    @staticmethod
    def export_to_excel(metrics=None):
        """Exportar portafolio a Excel"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja 1: Composición del portafolio
                portfolio_data = []
                for ticker, info in st.session_state.portfolio_assets.items():
                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                    portfolio_data.append({
                        'Ticker': ticker,
                        'Nombre': info['name'],
                        'Categoría': info['category'],
                        'Peso (%)': weight,
                        'Fecha Agregado': info['date_added'].strftime('%Y-%m-%d %H:%M')
                    })
                
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_df.to_excel(writer, sheet_name='Portafolio', index=False)
                
                # Hoja 2: Métricas
                if metrics:
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])
                    metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
                
                # Hoja 3: Asignación por categoría
                if portfolio_data:
                    category_df = portfolio_df.groupby('Categoría')['Peso (%)'].sum().reset_index()
                    category_df.to_excel(writer, sheet_name='Por Categoría', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error exportando: {e}")
            return None

def render_portfolio_sidebar():
    """Renderizar widget del portafolio en el sidebar"""
    PortfolioManager.initialize_portfolio_state()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Mi Portafolio")
    
    num_assets = len(st.session_state.portfolio_assets)
    
    if num_assets > 0:
        st.sidebar.success(f"📊 {num_assets} activo(s)")
        
        # Mostrar total de pesos
        total_weight = sum(st.session_state.portfolio_weights.values())
        st.sidebar.metric("Peso Total", f"{total_weight:.1f}%")
        
        # Botón para mostrar panel completo
        if st.sidebar.button("🔍 Gestionar Portafolio", use_container_width=True):
            st.session_state.show_portfolio_panel = True
            st.rerun()
        
        # Botón para limpiar
        if st.sidebar.button("🗑️ Limpiar Todo", use_container_width=True):
            st.session_state.portfolio_assets = {}
            st.session_state.portfolio_weights = {}
            st.success("🧹 Portafolio limpiado")
            st.rerun()
        
        # Lista rápida de activos
        st.sidebar.markdown("**Activos:**")
        for ticker, info in list(st.session_state.portfolio_assets.items())[:5]:
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
        
        if num_assets > 5:
            st.sidebar.caption(f"... y {num_assets - 5} más")
    else:
        st.sidebar.info("Carrito vacío")
        st.sidebar.caption("💡 Agrega fondos desde la tabla")

def render_add_button(ticker, fund_name, etf_dict):
    """Renderizar botón para agregar al portafolio"""
    PortfolioManager.initialize_portfolio_state()
    
    # Determinar categoría
    category = "General"
    if etf_dict is not None and not etf_dict.empty:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        if not fund_info.empty:
            for col in ['Category', 'Asset Class', 'Type']:
                if col in fund_info.columns:
                    category = fund_info[col].iloc[0]
                    break
    
    # Botón
    if ticker in st.session_state.portfolio_assets:
        return st.button("✅ En Portafolio", disabled=True, key=f"btn_{ticker}")
    else:
        if st.button("➕ Agregar", key=f"btn_{ticker}"):
            PortfolioManager.add_to_portfolio(ticker, fund_name, category)
            st.rerun()

def render_portfolio_panel(funds_data, etf_dict):
    """Renderizar panel completo de gestión de portafolios"""
    st.markdown("# 🛒 Gestión de Portafolio")
    
    if not st.session_state.portfolio_assets:
        st.info("Tu portafolio está vacío. Agrega fondos desde la tabla principal.")
        if st.button("← Volver al Análisis"):
            st.session_state.show_portfolio_panel = False
            st.rerun()
        return
    
    # Botón para volver
    if st.button("← Volver al Análisis"):
        st.session_state.show_portfolio_panel = False
        st.rerun()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## ⚖️ Ajustar Pesos")
        
        # Editor de pesos
        with st.form("portfolio_editor"):
            new_weights = {}
            new_categories = {}
            
            for ticker, info in st.session_state.portfolio_assets.items():
                current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                
                col_name, col_weight, col_cat, col_remove = st.columns([3, 2, 2, 1])
                
                with col_name:
                    st.write(f"**{info['name']}**")
                    st.caption(f"({ticker})")
                
                with col_weight:
                    new_weights[ticker] = st.number_input(
                        "Peso %",
                        min_value=0.0,
                        max_value=100.0,
                        value=current_weight,
                        step=0.1,
                        key=f"weight_{ticker}",
                        label_visibility="collapsed"
                    )
                
                with col_cat:
                    categories = ["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "General"]
                    current_cat = info.get('category', 'General')
                    try:
                        cat_index = categories.index(current_cat)
                    except ValueError:
                        cat_index = 4  # General
                    
                    new_categories[ticker] = st.selectbox(
                        "Categoría",
                        categories,
                        index=cat_index,
                        key=f"cat_{ticker}",
                        label_visibility="collapsed"
                    )
                
                with col_remove:
                    st.write("")  # Espaciado
                    if st.form_submit_button("🗑️", key=f"remove_{ticker}"):
                        PortfolioManager.remove_from_portfolio(ticker)
                        st.rerun()
            
            col_update, col_normalize = st.columns(2)
            
            with col_update:
                if st.form_submit_button("💾 Actualizar", use_container_width=True):
                    # Actualizar pesos y categorías
                    for ticker, weight in new_weights.items():
                        st.session_state.portfolio_weights[ticker] = weight
                    for ticker, category in new_categories.items():
                        st.session_state.portfolio_assets[ticker]['category'] = category
                    st.success("✅ Portafolio actualizado")
                    st.rerun()
            
            with col_normalize:
                if st.form_submit_button("⚖️ Normalizar 100%", use_container_width=True):
                    PortfolioManager.normalize_weights()
                    st.success("✅ Pesos normalizados")
                    st.rerun()
        
        # Verificar suma de pesos
        total_weight = sum(st.session_state.portfolio_weights.values())
        if abs(total_weight - 100) > 0.1:
            st.warning(f"⚠️ Los pesos suman {total_weight:.1f}%. Se recomienda normalizar a 100%.")
    
    with col2:
        st.markdown("## 📊 Visualización")
        
        # Gráfico de asignación
        allocation_chart = PortfolioManager.create_allocation_chart()
        if allocation_chart:
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        # Métricas básicas
        st.markdown("### 📈 Resumen")
        num_assets = len(st.session_state.portfolio_assets)
        total_weight = sum(st.session_state.portfolio_weights.values())
        
        st.metric("Activos", num_assets)
        st.metric("Peso Total", f"{total_weight:.1f}%")
        
        # Por categoría
        categories = {}
        for ticker, info in st.session_state.portfolio_assets.items():
            cat = info.get('category', 'General')
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            categories[cat] = categories.get(cat, 0) + weight
        
        st.markdown("### 🏷️ Por Categoría")
        for cat, weight in categories.items():
            st.metric(cat, f"{weight:.1f}%")
    
    # Análisis de Performance
    st.markdown("## 📈 Análisis de Performance")
    
    col_start, col_end, col_analyze = st.columns([2, 2, 1])
    
    with col_start:
        start_date = st.date_input(
            "Fecha Inicio",
            value=datetime.now().date() - timedelta(days=365),
            key="portfolio_start"
        )
    
    with col_end:
        end_date = st.date_input(
            "Fecha Fin",
            value=datetime.now().date(),
            key="portfolio_end"
        )
    
    with col_analyze:
        st.write("")  # Espaciado
        analyze_button = st.button("🔄 Analizar", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Calculando performance..."):
            metrics, chart_data = PortfolioManager.calculate_portfolio_performance(
                funds_data, pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
            
            if metrics and chart_data:
                # Mostrar métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Retorno Total", f"{metrics['Retorno Total (%)']:.2f}%")
                    st.metric("Volatilidad", f"{metrics['Volatilidad (%)']:.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
                
                with col3:
                    st.metric("VaR 5%", f"{metrics['VaR 5% (%)']:.2f}%")
                    st.metric("CVaR 5%", f"{metrics['CVaR 5% (%)']:.2f}%")
                
                # Gráfico de performance
                dates, values = chart_data
                chart = PortfolioManager.create_portfolio_chart(dates, values)
                st.plotly_chart(chart, use_container_width=True)
                
                # Botón de descarga
                st.markdown("### 💾 Exportar")
                excel_data = PortfolioManager.export_to_excel(metrics)
                if excel_data:
                    st.download_button(
                        label="📊 Descargar Excel",
                        data=excel_data,
                        file_name=f"portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            else:
                st.error("No se pudo calcular la performance. Verifica las fechas y que los fondos tengan datos.")

# Función principal modificada
def main():
    """Función principal del dashboard mejorado"""
    
    # Inicializar estado del portafolio
    PortfolioManager.initialize_portfolio_state()
    
    # Cargar datos usando la función original
    funds_data, etf_dict = load_data()
    
    if funds_data is None or etf_dict is None:
        return
    
    # Renderizar sidebar del portafolio
    render_portfolio_sidebar()
    
    # Verificar si mostrar panel de portafolio
    if st.session_state.get('show_portfolio_panel', False):
        render_portfolio_panel(funds_data, etf_dict)
        return
    
    # Continuar con la lógica original del dashboard pero con modificaciones mínimas
    # [Aquí iría el resto del código del dashboard original con pequeñas modificaciones]
    
    # Por ahora, mostrar mensaje de que se está ejecutando la versión mejorada
    st.markdown("# 📈 Dashboard de Análisis de Fondos (Versión Mejorada)")
    st.info("🚧 Esta es la versión mejorada con funcionalidad de portafolios. Funcionalidad completa en desarrollo...")
    
    # Mostrar tabla básica con botones de agregar
    st.markdown("## 🏆 Fondos Disponibles")
    
    # Obtener lista de fondos
    fund_columns = [col for col in funds_data.columns if col != 'Dates']
    
    # Mostrar algunos fondos de ejemplo con botones
    for i, ticker in enumerate(fund_columns[:10]):  # Mostrar primeros 10
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        fund_name = fund_info['Fund Name'].iloc[0] if not fund_info.empty and 'Fund Name' in fund_info.columns else ticker
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"**{fund_name}** ({ticker})")
        
        with col2:
            render_add_button(ticker, fund_name, etf_dict)

if __name__ == "__main__":
    main()