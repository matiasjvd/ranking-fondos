#!/usr/bin/env python3
"""
PORTFOLIO BUILDER MODULE
Módulo adicional para construcción y análisis de portafolios
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List, Tuple, Optional

class PortfolioBuilder:
    """Clase para manejar la construcción y análisis de portafolios"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Inicializar el estado de sesión para el portafolio"""
        if 'portfolio_assets' not in st.session_state:
            st.session_state.portfolio_assets = {}
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'portfolio_categories' not in st.session_state:
            st.session_state.portfolio_categories = {}
    
    def add_asset_to_portfolio(self, ticker: str, fund_name: str, category: str = "Sin categoría"):
        """Agregar un activo al portafolio"""
        st.session_state.portfolio_assets[ticker] = {
            'name': fund_name,
            'category': category,
            'added_date': datetime.now()
        }
        # Inicializar peso igual para todos los activos
        self._rebalance_weights()
        st.success(f"✅ {fund_name} agregado al portafolio")
    
    def remove_asset_from_portfolio(self, ticker: str):
        """Remover un activo del portafolio"""
        if ticker in st.session_state.portfolio_assets:
            fund_name = st.session_state.portfolio_assets[ticker]['name']
            del st.session_state.portfolio_assets[ticker]
            if ticker in st.session_state.portfolio_weights:
                del st.session_state.portfolio_weights[ticker]
            self._rebalance_weights()
            st.success(f"🗑️ {fund_name} removido del portafolio")
    
    def _rebalance_weights(self):
        """Rebalancear pesos para que sumen 100%"""
        num_assets = len(st.session_state.portfolio_assets)
        if num_assets > 0:
            equal_weight = 100.0 / num_assets
            for ticker in st.session_state.portfolio_assets.keys():
                st.session_state.portfolio_weights[ticker] = equal_weight
    
    def update_asset_weight(self, ticker: str, weight: float):
        """Actualizar el peso de un activo"""
        st.session_state.portfolio_weights[ticker] = weight
    
    def update_asset_category(self, ticker: str, category: str):
        """Actualizar la categoría de un activo"""
        st.session_state.portfolio_assets[ticker]['category'] = category
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Obtener resumen del portafolio actual"""
        if not st.session_state.portfolio_assets:
            return pd.DataFrame()
        
        data = []
        for ticker, asset_info in st.session_state.portfolio_assets.items():
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            data.append({
                'Ticker': ticker,
                'Nombre': asset_info['name'],
                'Categoría': asset_info['category'],
                'Peso (%)': weight,
                'Fecha Agregado': asset_info['added_date'].strftime('%Y-%m-%d %H:%M')
            })
        
        return pd.DataFrame(data)
    
    def normalize_weights(self):
        """Normalizar pesos para que sumen 100%"""
        total_weight = sum(st.session_state.portfolio_weights.values())
        if total_weight > 0:
            for ticker in st.session_state.portfolio_weights:
                st.session_state.portfolio_weights[ticker] = (
                    st.session_state.portfolio_weights[ticker] / total_weight * 100
                )
    
    def calculate_portfolio_metrics(self, funds_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Dict:
        """Calcular métricas del portafolio"""
        if not st.session_state.portfolio_assets:
            return {}
        
        try:
            # Filtrar datos por fechas
            funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
            filtered_data = funds_df[(funds_df['Dates'] >= start_date) & (funds_df['Dates'] <= end_date)].copy()
            
            if filtered_data.empty:
                return {}
            
            # Obtener tickers del portafolio
            portfolio_tickers = list(st.session_state.portfolio_assets.keys())
            available_tickers = [t for t in portfolio_tickers if t in filtered_data.columns]
            
            if not available_tickers:
                return {}
            
            # Calcular retornos de cada activo
            returns_data = {}
            portfolio_values = []
            dates = []
            
            for _, row in filtered_data.iterrows():
                portfolio_value = 0
                total_weight = 0
                
                for ticker in available_tickers:
                    if pd.notna(row[ticker]):
                        weight = st.session_state.portfolio_weights.get(ticker, 0) / 100
                        portfolio_value += row[ticker] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_values.append(portfolio_value / total_weight)
                    dates.append(row['Dates'])
            
            if len(portfolio_values) < 2:
                return {}
            
            # Convertir a Series para cálculos
            portfolio_series = pd.Series(portfolio_values, index=dates)
            returns = portfolio_series.pct_change().dropna()
            
            # Calcular métricas
            total_return = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1) * 100
            
            # Volatilidad anualizada
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe Ratio (asumiendo risk-free rate = 0)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Max Drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            # VaR y CVaR al 5%
            var_5 = np.percentile(returns, 5) * np.sqrt(252) * 100
            cvar_5 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(252) * 100
            
            return {
                'Retorno Total (%)': total_return,
                'Volatilidad Anualizada (%)': volatility,
                'Sharpe Ratio': sharpe_ratio,
                'Max Drawdown (%)': max_drawdown,
                'VaR 5% Anualizado (%)': var_5,
                'CVaR 5% Anualizado (%)': cvar_5,
                'Número de Activos': len(available_tickers),
                'Período': f"{start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            st.error(f"Error calculando métricas del portafolio: {e}")
            return {}
    
    def create_portfolio_performance_chart(self, funds_df: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Crear gráfico de performance del portafolio"""
        if not st.session_state.portfolio_assets:
            return None
        
        try:
            # Filtrar datos por fechas
            funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
            filtered_data = funds_df[(funds_df['Dates'] >= start_date) & (funds_df['Dates'] <= end_date)].copy()
            
            if filtered_data.empty:
                return None
            
            # Obtener tickers del portafolio
            portfolio_tickers = list(st.session_state.portfolio_assets.keys())
            available_tickers = [t for t in portfolio_tickers if t in filtered_data.columns]
            
            if not available_tickers:
                return None
            
            # Calcular valor del portafolio
            portfolio_values = []
            dates = []
            
            for _, row in filtered_data.iterrows():
                portfolio_value = 0
                total_weight = 0
                
                for ticker in available_tickers:
                    if pd.notna(row[ticker]):
                        weight = st.session_state.portfolio_weights.get(ticker, 0) / 100
                        portfolio_value += row[ticker] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_values.append(portfolio_value / total_weight)
                    dates.append(row['Dates'])
            
            if len(portfolio_values) < 2:
                return None
            
            # Normalizar a base 100
            base_value = portfolio_values[0]
            normalized_values = [(v / base_value) * 100 for v in portfolio_values]
            
            # Crear gráfico
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=normalized_values,
                mode='lines',
                name='Portafolio',
                line=dict(color='#60a5fa', width=3),
                hovertemplate='<b>Portafolio</b><br>' +
                            'Fecha: %{x}<br>' +
                            'Valor: %{y:.2f}<br>' +
                            '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Performance del Portafolio<br><sub>Base 100 - {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}</sub>",
                xaxis_title="Fecha",
                yaxis_title="Valor (Base 100)",
                hovermode='x unified',
                height=500,
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
            st.error(f"Error creando gráfico del portafolio: {e}")
            return None
    
    def create_allocation_chart(self):
        """Crear gráfico de asignación del portafolio"""
        if not st.session_state.portfolio_assets:
            return None
        
        try:
            # Preparar datos para el gráfico
            tickers = list(st.session_state.portfolio_assets.keys())
            names = [st.session_state.portfolio_assets[t]['name'] for t in tickers]
            weights = [st.session_state.portfolio_weights.get(t, 0) for t in tickers]
            categories = [st.session_state.portfolio_assets[t]['category'] for t in tickers]
            
            # Crear gráfico de pie
            fig = go.Figure(data=[go.Pie(
                labels=[f"{name}<br>({ticker})" for name, ticker in zip(names, tickers)],
                values=weights,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='#000000', width=2)
                )
            )])
            
            fig.update_layout(
                title="Asignación del Portafolio",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                title_font=dict(color='#fafafa', size=16)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creando gráfico de asignación: {e}")
            return None
    
    def export_portfolio_to_excel(self, metrics: Dict = None) -> bytes:
        """Exportar portafolio a Excel"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja 1: Resumen del portafolio
                portfolio_df = self.get_portfolio_summary()
                if not portfolio_df.empty:
                    portfolio_df.to_excel(writer, sheet_name='Portafolio', index=False)
                
                # Hoja 2: Métricas
                if metrics:
                    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])
                    metrics_df.to_excel(writer, sheet_name='Métricas', index=False)
                
                # Hoja 3: Asignación por categoría
                if not portfolio_df.empty:
                    category_allocation = portfolio_df.groupby('Categoría')['Peso (%)'].sum().reset_index()
                    category_allocation.to_excel(writer, sheet_name='Asignación por Categoría', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error exportando a Excel: {e}")
            return b""
    
    def clear_portfolio(self):
        """Limpiar todo el portafolio"""
        st.session_state.portfolio_assets = {}
        st.session_state.portfolio_weights = {}
        st.session_state.portfolio_categories = {}
        st.success("🧹 Portafolio limpiado")

def render_portfolio_cart_widget():
    """Renderizar widget del carrito de portafolio en el sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🛒 Carrito de Portafolio")
    
    portfolio_builder = PortfolioBuilder()
    
    # Mostrar número de activos en el carrito
    num_assets = len(st.session_state.portfolio_assets)
    if num_assets > 0:
        st.sidebar.success(f"📊 {num_assets} activo(s) en el portafolio")
        
        # Botón para ir a la pestaña de portafolio
        if st.sidebar.button("🔍 Ver Portafolio Completo", use_container_width=True):
            st.session_state.show_portfolio_tab = True
            st.rerun()
        
        # Mostrar resumen rápido
        total_weight = sum(st.session_state.portfolio_weights.values())
        st.sidebar.metric("Peso Total", f"{total_weight:.1f}%")
        
        # Botón para limpiar
        if st.sidebar.button("🗑️ Limpiar Portafolio", use_container_width=True):
            portfolio_builder.clear_portfolio()
            st.rerun()
    else:
        st.sidebar.info("El carrito está vacío")
        st.sidebar.markdown("💡 *Agrega fondos desde la tabla principal*")

def render_add_to_portfolio_button(ticker: str, fund_name: str, etf_dict: pd.DataFrame = None):
    """Renderizar botón para agregar fondo al portafolio"""
    portfolio_builder = PortfolioBuilder()
    
    # Determinar categoría del fondo
    category = "Sin categoría"
    if etf_dict is not None and not etf_dict.empty:
        fund_info = etf_dict[etf_dict['Ticker'] == ticker]
        if not fund_info.empty:
            # Intentar obtener categoría de las columnas disponibles
            if 'Category' in fund_info.columns:
                category = fund_info['Category'].iloc[0]
            elif 'Asset Class' in fund_info.columns:
                category = fund_info['Asset Class'].iloc[0]
            elif 'Type' in fund_info.columns:
                category = fund_info['Type'].iloc[0]
    
    # Verificar si ya está en el portafolio
    if ticker in st.session_state.portfolio_assets:
        return st.button("✅ En Portafolio", disabled=True, key=f"portfolio_{ticker}")
    else:
        if st.button("➕ Agregar", key=f"portfolio_{ticker}"):
            portfolio_builder.add_asset_to_portfolio(ticker, fund_name, category)
            st.rerun()

def render_portfolio_management_tab(funds_data: pd.DataFrame, etf_dict: pd.DataFrame):
    """Renderizar la pestaña completa de gestión de portafolios"""
    st.markdown("# 🛒 Constructor de Portafolios")
    st.markdown("Construye y analiza tu portafolio personalizado de fondos")
    
    portfolio_builder = PortfolioBuilder()
    
    # Verificar si hay activos en el portafolio
    if not st.session_state.portfolio_assets:
        st.info("🔍 Tu portafolio está vacío. Ve a la pestaña principal para agregar fondos.")
        return
    
    # Configuración de columnas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📊 Gestión del Portafolio")
        
        # Obtener resumen del portafolio
        portfolio_df = portfolio_builder.get_portfolio_summary()
        
        if not portfolio_df.empty:
            # Editor de pesos
            st.markdown("### ⚖️ Ajustar Pesos")
            
            # Crear formulario para editar pesos
            with st.form("weight_editor"):
                new_weights = {}
                categories = {}
                
                for _, row in portfolio_df.iterrows():
                    ticker = row['Ticker']
                    current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                    
                    col_weight, col_category, col_remove = st.columns([2, 2, 1])
                    
                    with col_weight:
                        new_weights[ticker] = st.number_input(
                            f"Peso {row['Nombre']} (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=current_weight,
                            step=0.1,
                            key=f"weight_{ticker}"
                        )
                    
                    with col_category:
                        current_category = st.session_state.portfolio_assets[ticker]['category']
                        categories[ticker] = st.selectbox(
                            f"Categoría {row['Nombre']}",
                            options=["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "Sin categoría"],
                            index=["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "Sin categoría"].index(current_category) if current_category in ["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "Sin categoría"] else 4,
                            key=f"category_{ticker}"
                        )
                    
                    with col_remove:
                        st.markdown("<br>", unsafe_allow_html=True)  # Espaciado
                        if st.form_submit_button(f"🗑️", key=f"remove_{ticker}"):
                            portfolio_builder.remove_asset_from_portfolio(ticker)
                            st.rerun()
                
                col_submit, col_normalize = st.columns(2)
                
                with col_submit:
                    if st.form_submit_button("💾 Actualizar Pesos", use_container_width=True):
                        # Actualizar pesos y categorías
                        for ticker, weight in new_weights.items():
                            portfolio_builder.update_asset_weight(ticker, weight)
                        for ticker, category in categories.items():
                            portfolio_builder.update_asset_category(ticker, category)
                        st.success("✅ Pesos y categorías actualizados")
                        st.rerun()
                
                with col_normalize:
                    if st.form_submit_button("⚖️ Normalizar a 100%", use_container_width=True):
                        portfolio_builder.normalize_weights()
                        st.success("✅ Pesos normalizados")
                        st.rerun()
            
            # Mostrar tabla actualizada
            st.markdown("### 📋 Resumen Actual")
            updated_portfolio_df = portfolio_builder.get_portfolio_summary()
            st.dataframe(updated_portfolio_df, use_container_width=True, hide_index=True)
            
            # Verificar suma de pesos
            total_weight = sum(st.session_state.portfolio_weights.values())
            if abs(total_weight - 100) > 0.1:
                st.warning(f"⚠️ Los pesos suman {total_weight:.1f}%. Se recomienda que sumen 100%.")
    
    with col2:
        st.markdown("## 📈 Visualizaciones")
        
        # Gráfico de asignación
        allocation_chart = portfolio_builder.create_allocation_chart()
        if allocation_chart:
            st.plotly_chart(allocation_chart, use_container_width=True)
        
        # Métricas rápidas
        st.markdown("### 📊 Métricas Rápidas")
        num_assets = len(st.session_state.portfolio_assets)
        total_weight = sum(st.session_state.portfolio_weights.values())
        
        st.metric("Número de Activos", num_assets)
        st.metric("Peso Total", f"{total_weight:.1f}%")
        
        # Distribución por categoría
        if not portfolio_df.empty:
            category_dist = portfolio_df.groupby('Categoría')['Peso (%)'].sum()
            st.markdown("### 🏷️ Por Categoría")
            for category, weight in category_dist.items():
                st.metric(category, f"{weight:.1f}%")
    
    # Análisis de Performance
    st.markdown("## 📈 Análisis de Performance")
    
    # Selector de fechas para análisis
    col_start, col_end = st.columns(2)
    
    with col_start:
        analysis_start_date = st.date_input(
            "Fecha de Inicio para Análisis",
            value=datetime.now().date() - timedelta(days=365),
            key="portfolio_start_date"
        )
    
    with col_end:
        analysis_end_date = st.date_input(
            "Fecha de Fin para Análisis",
            value=datetime.now().date(),
            key="portfolio_end_date"
        )
    
    if st.button("🔄 Calcular Métricas", use_container_width=True):
        with st.spinner("Calculando métricas del portafolio..."):
            # Calcular métricas
            metrics = portfolio_builder.calculate_portfolio_metrics(
                funds_data, 
                pd.to_datetime(analysis_start_date), 
                pd.to_datetime(analysis_end_date)
            )
            
            if metrics:
                # Mostrar métricas en columnas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Retorno Total", f"{metrics.get('Retorno Total (%)', 0):.2f}%")
                    st.metric("Volatilidad", f"{metrics.get('Volatilidad Anualizada (%)', 0):.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")
                    st.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
                
                with col3:
                    st.metric("VaR 5%", f"{metrics.get('VaR 5% Anualizado (%)', 0):.2f}%")
                    st.metric("CVaR 5%", f"{metrics.get('CVaR 5% Anualizado (%)', 0):.2f}%")
                
                # Gráfico de performance
                performance_chart = portfolio_builder.create_portfolio_performance_chart(
                    funds_data,
                    pd.to_datetime(analysis_start_date),
                    pd.to_datetime(analysis_end_date)
                )
                
                if performance_chart:
                    st.plotly_chart(performance_chart, use_container_width=True)
                
                # Botón de descarga
                st.markdown("## 💾 Exportar Resultados")
                
                excel_data = portfolio_builder.export_portfolio_to_excel(metrics)
                if excel_data:
                    st.download_button(
                        label="📊 Descargar Portafolio (Excel)",
                        data=excel_data,
                        file_name=f"portafolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            else:
                st.error("No se pudieron calcular las métricas. Verifica que los fondos tengan datos en el período seleccionado.")