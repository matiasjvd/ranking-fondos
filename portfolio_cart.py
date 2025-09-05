#!/usr/bin/env python3
"""
PORTFOLIO CART MODULE
Módulo de carrito de portafolios que se integra al dashboard original
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import io

class PortfolioCart:
    """Clase para manejar el carrito de portafolios"""
    
    @staticmethod
    def initialize():
        """Inicializar el estado del carrito"""
        if 'portfolio_cart' not in st.session_state:
            st.session_state.portfolio_cart = {}
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'show_portfolio_tab' not in st.session_state:
            st.session_state.show_portfolio_tab = False
    
    @staticmethod
    def add_to_cart(ticker, fund_name, category="General"):
        """Agregar fondo al carrito"""
        PortfolioCart.initialize()
        
        if ticker not in st.session_state.portfolio_cart:
            st.session_state.portfolio_cart[ticker] = {
                'name': fund_name,
                'category': category,
                'date_added': datetime.now()
            }
            PortfolioCart._rebalance_weights()
            st.success(f"✅ {fund_name} agregado al carrito")
            return True
        else:
            st.warning(f"⚠️ {fund_name} ya está en el carrito")
            return False
    
    @staticmethod
    def remove_from_cart(ticker):
        """Remover fondo del carrito"""
        if ticker in st.session_state.portfolio_cart:
            fund_name = st.session_state.portfolio_cart[ticker]['name']
            del st.session_state.portfolio_cart[ticker]
            if ticker in st.session_state.portfolio_weights:
                del st.session_state.portfolio_weights[ticker]
            PortfolioCart._rebalance_weights()
            st.success(f"🗑️ {fund_name} removido del carrito")
            return True
        return False
    
    @staticmethod
    def clear_cart():
        """Limpiar todo el carrito"""
        st.session_state.portfolio_cart = {}
        st.session_state.portfolio_weights = {}
        st.success("🧹 Carrito limpiado")
    
    @staticmethod
    def _rebalance_weights():
        """Rebalancear pesos equitativamente"""
        num_items = len(st.session_state.portfolio_cart)
        if num_items > 0:
            equal_weight = 100.0 / num_items
            for ticker in st.session_state.portfolio_cart.keys():
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
    def get_cart_summary():
        """Obtener resumen del carrito"""
        PortfolioCart.initialize()
        num_items = len(st.session_state.portfolio_cart)
        total_weight = sum(st.session_state.portfolio_weights.values())
        return {
            'num_items': num_items,
            'total_weight': total_weight,
            'items': st.session_state.portfolio_cart,
            'weights': st.session_state.portfolio_weights
        }
    
    @staticmethod
    def render_cart_widget():
        """Renderizar widget del carrito en el sidebar"""
        PortfolioCart.initialize()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🛒 Mi Carrito de Portafolio")
        
        summary = PortfolioCart.get_cart_summary()
        
        if summary['num_items'] > 0:
            # Mostrar resumen
            st.sidebar.success(f"📊 {summary['num_items']} fondo(s) en carrito")
            st.sidebar.metric("Peso Total", f"{summary['total_weight']:.1f}%")
            
            # Lista de fondos en el carrito
            st.sidebar.markdown("**Fondos en carrito:**")
            for ticker, info in list(summary['items'].items())[:5]:  # Mostrar máximo 5
                weight = summary['weights'].get(ticker, 0)
                col1, col2 = st.sidebar.columns([3, 1])
                with col1:
                    st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
                with col2:
                    if st.sidebar.button("🗑️", key=f"remove_sidebar_{ticker}", help="Remover"):
                        PortfolioCart.remove_from_cart(ticker)
                        st.rerun()
            
            if summary['num_items'] > 5:
                st.sidebar.caption(f"... y {summary['num_items'] - 5} más")
            
            # Botones de acción
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📊 Ver Portafolio", use_container_width=True):
                    st.session_state.show_portfolio_tab = True
                    st.rerun()
            
            with col2:
                if st.button("🧹 Limpiar", use_container_width=True):
                    PortfolioCart.clear_cart()
                    st.rerun()
        else:
            st.sidebar.info("🛒 Carrito vacío")
            st.sidebar.caption("💡 Agrega fondos desde la tabla principal")
    
    @staticmethod
    def render_add_button(ticker, fund_name, key_suffix=""):
        """Renderizar botón para agregar al carrito"""
        PortfolioCart.initialize()
        
        if ticker in st.session_state.portfolio_cart:
            return st.button(
                "✅ En Carrito", 
                disabled=True, 
                key=f"add_btn_{ticker}_{key_suffix}",
                help="Ya está en el carrito"
            )
        else:
            if st.button(
                "🛒 Agregar", 
                key=f"add_btn_{ticker}_{key_suffix}",
                help=f"Agregar {fund_name} al carrito"
            ):
                PortfolioCart.add_to_cart(ticker, fund_name)
                st.rerun()
                return True
        return False
    
    @staticmethod
    def calculate_portfolio_performance(funds_df, start_date, end_date):
        """Calcular performance del portafolio"""
        PortfolioCart.initialize()
        
        if not st.session_state.portfolio_cart:
            return None
        
        try:
            # Filtrar datos por fechas
            funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
            filtered_data = funds_df[
                (funds_df['Dates'] >= start_date) & 
                (funds_df['Dates'] <= end_date)
            ].copy()
            
            if filtered_data.empty:
                return None
            
            # Calcular valor del portafolio día a día
            portfolio_values = []
            dates = []
            
            for _, row in filtered_data.iterrows():
                portfolio_value = 0
                total_weight = 0
                
                for ticker in st.session_state.portfolio_cart.keys():
                    if ticker in filtered_data.columns and pd.notna(row[ticker]):
                        weight = st.session_state.portfolio_weights.get(ticker, 0) / 100
                        portfolio_value += row[ticker] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_values.append(portfolio_value / total_weight)
                    dates.append(row['Dates'])
            
            if len(portfolio_values) < 2:
                return None
            
            # Calcular retornos
            portfolio_series = pd.Series(portfolio_values, index=dates)
            returns = portfolio_series.pct_change().dropna()
            
            # Métricas
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
            
            # Normalizar para gráfico (base 100)
            base_value = portfolio_values[0]
            normalized_values = [(v / base_value) * 100 for v in portfolio_values]
            
            return {
                'metrics': {
                    'Retorno Total (%)': total_return,
                    'Volatilidad Anualizada (%)': volatility,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown (%)': max_drawdown,
                    'VaR 5% (%)': var_5,
                    'CVaR 5% (%)': cvar_5
                },
                'chart_data': {
                    'dates': dates,
                    'values': normalized_values
                }
            }
            
        except Exception as e:
            st.error(f"Error calculando performance: {e}")
            return None
    
    @staticmethod
    def create_portfolio_chart(chart_data):
        """Crear gráfico del portafolio"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=chart_data['dates'],
            y=chart_data['values'],
            mode='lines',
            name='Mi Portafolio',
            line=dict(color='#60a5fa', width=3),
            hovertemplate='<b>Mi Portafolio</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Performance de Mi Portafolio (Base 100)",
            xaxis_title="Fecha",
            yaxis_title="Valor Normalizado",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#404040', color='#fafafa'),
            yaxis=dict(gridcolor='#404040', color='#fafafa'),
            title_font=dict(size=16, color='#fafafa')
        )
        
        return fig
    
    @staticmethod
    def create_allocation_chart():
        """Crear gráfico de asignación del portafolio"""
        PortfolioCart.initialize()
        
        if not st.session_state.portfolio_cart:
            return None
        
        tickers = list(st.session_state.portfolio_cart.keys())
        names = [st.session_state.portfolio_cart[t]['name'] for t in tickers]
        weights = [st.session_state.portfolio_weights.get(t, 0) for t in tickers]
        
        # Crear etiquetas más informativas
        labels = []
        for i, (name, ticker, weight) in enumerate(zip(names, tickers, weights)):
            # Truncar nombre si es muy largo
            display_name = name[:20] + "..." if len(name) > 20 else name
            labels.append(f"{display_name}<br>({ticker})<br>{weight:.1f}%")
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=weights,
            hole=0.4,
            textinfo='label',
            textposition='outside',
            marker=dict(
                colors=px.colors.qualitative.Set3,
                line=dict(color='#000000', width=2)
            )
        )])
        
        fig.update_layout(
            title="Asignación de Mi Portafolio",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa', size=12),
            title_font=dict(size=16, color='#fafafa'),
            showlegend=False  # Las etiquetas ya están en el gráfico
        )
        
        return fig
    
    @staticmethod
    def export_portfolio_to_excel():
        """Exportar portafolio a Excel"""
        PortfolioCart.initialize()
        
        if not st.session_state.portfolio_cart:
            return None
        
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja 1: Composición del portafolio
                portfolio_data = []
                for ticker, info in st.session_state.portfolio_cart.items():
                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                    portfolio_data.append({
                        'Ticker': ticker,
                        'Nombre del Fondo': info['name'],
                        'Categoría': info.get('category', 'General'),
                        'Peso (%)': round(weight, 2),
                        'Fecha Agregado': info['date_added'].strftime('%Y-%m-%d %H:%M:%S')
                    })
                
                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_df.to_excel(writer, sheet_name='Composición Portafolio', index=False)
                
                # Hoja 2: Resumen por categoría
                if portfolio_data:
                    category_summary = portfolio_df.groupby('Categoría')['Peso (%)'].sum().reset_index()
                    category_summary.to_excel(writer, sheet_name='Resumen por Categoría', index=False)
                
                # Hoja 3: Información del portafolio
                summary_data = {
                    'Métrica': [
                        'Número de Fondos',
                        'Peso Total (%)',
                        'Fecha de Creación',
                        'Última Modificación'
                    ],
                    'Valor': [
                        len(st.session_state.portfolio_cart),
                        round(sum(st.session_state.portfolio_weights.values()), 2),
                        min([info['date_added'] for info in st.session_state.portfolio_cart.values()]).strftime('%Y-%m-%d %H:%M:%S'),
                        max([info['date_added'] for info in st.session_state.portfolio_cart.values()]).strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Información General', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error exportando portafolio: {e}")
            return None
    
    @staticmethod
    def render_portfolio_management_tab():
        """Renderizar pestaña completa de gestión de portafolios"""
        PortfolioCart.initialize()
        
        st.markdown("# 🛒 Gestión de Mi Portafolio")
        
        # Botón para volver
        if st.button("← Volver al Análisis de Fondos"):
            st.session_state.show_portfolio_tab = False
            st.rerun()
        
        summary = PortfolioCart.get_cart_summary()
        
        if summary['num_items'] == 0:
            st.info("🛒 Tu carrito está vacío. Ve a la pestaña de análisis para agregar fondos.")
            return
        
        # Layout en columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## ⚖️ Ajustar Pesos de Activos")
            
            # Editor de pesos
            with st.form("portfolio_weights_form"):
                st.markdown("**Configura el peso de cada activo en tu portafolio:**")
                
                new_weights = {}
                new_categories = {}
                
                for ticker, info in st.session_state.portfolio_cart.items():
                    current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                    
                    # Crear fila para cada activo
                    col_name, col_weight, col_category, col_remove = st.columns([3, 1.5, 1.5, 1])
                    
                    with col_name:
                        st.markdown(f"**{info['name']}**")
                        st.caption(f"Ticker: {ticker}")
                    
                    with col_weight:
                        new_weights[ticker] = st.number_input(
                            "Peso %",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(current_weight),
                            step=0.5,
                            key=f"weight_input_{ticker}",
                            label_visibility="collapsed"
                        )
                    
                    with col_category:
                        categories = ["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "General"]
                        current_cat = info.get('category', 'General')
                        try:
                            cat_index = categories.index(current_cat)
                        except ValueError:
                            cat_index = 4  # General por defecto
                        
                        new_categories[ticker] = st.selectbox(
                            "Categoría",
                            categories,
                            index=cat_index,
                            key=f"category_select_{ticker}",
                            label_visibility="collapsed"
                        )
                    
                    with col_remove:
                        st.write("")  # Espaciado
                        if st.form_submit_button("🗑️", key=f"remove_form_{ticker}", help="Remover del carrito"):
                            PortfolioCart.remove_from_cart(ticker)
                            st.rerun()
                
                # Botones de acción
                st.markdown("---")
                col_update, col_normalize, col_equal = st.columns(3)
                
                with col_update:
                    if st.form_submit_button("💾 Actualizar Pesos", use_container_width=True):
                        # Actualizar pesos
                        for ticker, weight in new_weights.items():
                            st.session_state.portfolio_weights[ticker] = weight
                        
                        # Actualizar categorías
                        for ticker, category in new_categories.items():
                            if ticker in st.session_state.portfolio_cart:
                                st.session_state.portfolio_cart[ticker]['category'] = category
                        
                        st.success("✅ Portafolio actualizado correctamente")
                        st.rerun()
                
                with col_normalize:
                    if st.form_submit_button("⚖️ Normalizar a 100%", use_container_width=True):
                        PortfolioCart.normalize_weights()
                        st.success("✅ Pesos normalizados a 100%")
                        st.rerun()
                
                with col_equal:
                    if st.form_submit_button("🔄 Pesos Iguales", use_container_width=True):
                        PortfolioCart._rebalance_weights()
                        st.success("✅ Pesos distribuidos equitativamente")
                        st.rerun()
            
            # Verificación de suma de pesos
            total_weight = sum(st.session_state.portfolio_weights.values())
            if abs(total_weight - 100) > 0.1:
                if total_weight > 100:
                    st.warning(f"⚠️ Los pesos suman {total_weight:.1f}% (más de 100%). Se recomienda normalizar.")
                else:
                    st.info(f"ℹ️ Los pesos suman {total_weight:.1f}% (menos de 100%). Puedes ajustar o normalizar.")
            else:
                st.success(f"✅ Los pesos suman {total_weight:.1f}% - Portafolio balanceado")
        
        with col2:
            st.markdown("## 📊 Visualización")
            
            # Métricas básicas
            st.markdown("### 📈 Resumen del Portafolio")
            col_assets, col_weight = st.columns(2)
            with col_assets:
                st.metric("Número de Activos", summary['num_items'])
            with col_weight:
                st.metric("Peso Total", f"{summary['total_weight']:.1f}%")
            
            # Gráfico de asignación
            allocation_chart = PortfolioCart.create_allocation_chart()
            if allocation_chart:
                st.plotly_chart(allocation_chart, use_container_width=True)
            
            # Resumen por categoría
            st.markdown("### 🏷️ Por Categoría")
            categories = {}
            for ticker, info in st.session_state.portfolio_cart.items():
                cat = info.get('category', 'General')
                weight = st.session_state.portfolio_weights.get(ticker, 0)
                categories[cat] = categories.get(cat, 0) + weight
            
            for cat, weight in categories.items():
                st.metric(cat, f"{weight:.1f}%")
        
        # Análisis de Performance
        st.markdown("## 📈 Análisis de Performance Histórica")
        
        col_start, col_end, col_analyze = st.columns([2, 2, 1])
        
        with col_start:
            start_date = st.date_input(
                "📅 Fecha de Inicio",
                value=datetime.now().date() - timedelta(days=365),
                key="portfolio_analysis_start"
            )
        
        with col_end:
            end_date = st.date_input(
                "📅 Fecha de Fin",
                value=datetime.now().date(),
                key="portfolio_analysis_end"
            )
        
        with col_analyze:
            st.write("")  # Espaciado
            analyze_button = st.button("🔄 Analizar Performance", use_container_width=True)
        
        if analyze_button:
            # Necesitamos cargar los datos de fondos para el análisis
            # Esto requiere acceso a la función load_data del dashboard original
            st.info("💡 Para el análisis de performance, necesitamos cargar los datos de fondos...")
            
            # Placeholder para mostrar que la funcionalidad está disponible
            st.markdown("### 📊 Métricas Calculadas")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Retorno Total", "Calculando...", help="Retorno total del período")
                st.metric("Volatilidad", "Calculando...", help="Volatilidad anualizada")
            
            with col2:
                st.metric("Sharpe Ratio", "Calculando...", help="Retorno ajustado por riesgo")
                st.metric("Max Drawdown", "Calculando...", help="Máxima pérdida desde un pico")
            
            with col3:
                st.metric("VaR 5%", "Calculando...", help="Value at Risk al 5%")
                st.metric("CVaR 5%", "Calculando...", help="Conditional VaR al 5%")
            
            st.info("🔧 Funcionalidad de análisis completa disponible cuando se integre con los datos del dashboard principal.")
        
        # Exportación
        st.markdown("## 💾 Exportar Portafolio")
        
        col_export, col_info = st.columns([1, 2])
        
        with col_export:
            excel_data = PortfolioCart.export_portfolio_to_excel()
            if excel_data:
                st.download_button(
                    label="📊 Descargar Portafolio (Excel)",
                    data=excel_data,
                    file_name=f"mi_portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col_info:
            st.info("📋 El archivo Excel incluye: composición del portafolio, resumen por categorías e información general.")

# Función para integrar el carrito en cualquier dashboard
def integrate_portfolio_cart():
    """Función para integrar el carrito en el dashboard existente"""
    PortfolioCart.initialize()
    PortfolioCart.render_cart_widget()
    
    # Si se debe mostrar la pestaña de portafolio
    if st.session_state.get('show_portfolio_tab', False):
        PortfolioCart.render_portfolio_management_tab()
        return True  # Indica que se está mostrando la pestaña de portafolio
    
    return False  # Indica que se debe mostrar el contenido normal del dashboard