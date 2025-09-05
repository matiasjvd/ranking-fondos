#!/usr/bin/env python3
"""
SIMPLE PORTFOLIO CART MODULE
Módulo de carrito simple que se integra sin modificar el dashboard original
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import io
import cvxpy as cp

class SimpleCart:
    """Clase para manejar el carrito de portafolios simple"""
    
    @staticmethod
    def initialize():
        """Inicializar el estado del carrito"""
        if 'selected_funds' not in st.session_state:
            st.session_state.selected_funds = set()
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'show_cart_analysis' not in st.session_state:
            st.session_state.show_cart_analysis = False
    
    @staticmethod
    def render_fund_selector(ticker, fund_name, key_suffix=""):
        """Renderizar checkbox simple para seleccionar fondos"""
        SimpleCart.initialize()
        
        is_selected = ticker in st.session_state.selected_funds
        
        if st.checkbox(
            f"Seleccionar {fund_name}",
            value=is_selected,
            key=f"select_{ticker}_{key_suffix}",
            help=f"Agregar {fund_name} ({ticker}) al carrito"
        ):
            if ticker not in st.session_state.selected_funds:
                st.session_state.selected_funds.add(ticker)
                SimpleCart._rebalance_weights()
        else:
            if ticker in st.session_state.selected_funds:
                st.session_state.selected_funds.discard(ticker)
                if ticker in st.session_state.portfolio_weights:
                    del st.session_state.portfolio_weights[ticker]
                SimpleCart._rebalance_weights()
    
    @staticmethod
    def _rebalance_weights():
        """Rebalancear pesos equitativamente"""
        num_funds = len(st.session_state.selected_funds)
        if num_funds > 0:
            equal_weight = 100.0 / num_funds
            for ticker in st.session_state.selected_funds:
                st.session_state.portfolio_weights[ticker] = equal_weight
    
    @staticmethod
    def render_cart_sidebar():
        """Renderizar widget del carrito en el sidebar"""
        SimpleCart.initialize()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🛒 Mi Carrito")
        
        num_selected = len(st.session_state.selected_funds)
        
        if num_selected > 0:
            st.sidebar.success(f"📊 {num_selected} fondo(s) seleccionado(s)")
            
            # Mostrar fondos seleccionados
            for ticker in list(st.session_state.selected_funds)[:5]:  # Mostrar máximo 5
                weight = st.session_state.portfolio_weights.get(ticker, 0)
                st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
            
            if num_selected > 5:
                st.sidebar.caption(f"... y {num_selected - 5} más")
            
            # Botones de acción
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("📊 Analizar Carrito", use_container_width=True):
                    st.session_state.show_cart_analysis = True
                    st.rerun()
            
            with col2:
                if st.button("🧹 Limpiar", use_container_width=True):
                    st.session_state.selected_funds.clear()
                    st.session_state.portfolio_weights.clear()
                    st.rerun()
        else:
            st.sidebar.info("🛒 Carrito vacío")
            st.sidebar.caption("💡 Selecciona fondos desde la tabla principal")
    
    @staticmethod
    def calculate_portfolio_metrics(funds_df, start_date, end_date):
        """Calcular métricas del portafolio"""
        SimpleCart.initialize()
        
        if not st.session_state.selected_funds:
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
                
                for ticker in st.session_state.selected_funds:
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
            
            # Métricas básicas
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
                },
                'returns': returns
            }
            
        except Exception as e:
            st.error(f"Error calculando métricas: {e}")
            return None
    
    @staticmethod
    def calculate_individual_fund_metrics(funds_df, ticker):
        """Calcular métricas de un fondo individual (igual que el dashboard original)"""
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
    
    @staticmethod
    def calculate_efficient_frontier(funds_df, selected_funds, start_date, end_date):
        """Calcular frontera eficiente para fondos seleccionados"""
        try:
            # Filtrar datos
            funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
            filtered_data = funds_df[
                (funds_df['Dates'] >= start_date) & 
                (funds_df['Dates'] <= end_date)
            ].copy()
            
            if filtered_data.empty or len(selected_funds) < 2:
                return None
            
            # Calcular retornos para fondos seleccionados
            returns_data = {}
            for ticker in selected_funds:
                if ticker in filtered_data.columns:
                    prices = filtered_data[['Dates', ticker]].dropna()
                    if len(prices) > 1:
                        prices = prices.sort_values('Dates')
                        returns = prices[ticker].pct_change().dropna()
                        returns_data[ticker] = returns
            
            if len(returns_data) < 2:
                return None
            
            # Crear DataFrame de retornos
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 10:  # Necesitamos suficientes datos
                return None
            
            # Calcular matriz de covarianza y retornos esperados
            cov_matrix = returns_df.cov() * 252  # Anualizar
            expected_returns = returns_df.mean() * 252  # Anualizar
            
            n_assets = len(expected_returns)
            
            # Generar puntos de la frontera eficiente
            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), 50)
            efficient_portfolios = []
            
            for target in target_returns:
                try:
                    # Variables de optimización
                    weights = cp.Variable(n_assets)
                    
                    # Función objetivo: minimizar varianza
                    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
                    
                    # Restricciones
                    constraints = [
                        cp.sum(weights) == 1,  # Suma de pesos = 1
                        weights >= 0,  # No short selling
                        expected_returns.values @ weights == target  # Retorno objetivo
                    ]
                    
                    # Resolver optimización
                    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                    problem.solve(solver=cp.ECOS, verbose=False)
                    
                    if problem.status == cp.OPTIMAL:
                        portfolio_return = target
                        portfolio_risk = np.sqrt(problem.value)
                        efficient_portfolios.append({
                            'return': portfolio_return * 100,
                            'risk': portfolio_risk * 100,
                            'weights': weights.value
                        })
                
                except:
                    continue
            
            return {
                'portfolios': efficient_portfolios,
                'assets': list(expected_returns.index),
                'expected_returns': expected_returns * 100,
                'risks': np.sqrt(np.diag(cov_matrix)) * 100
            }
            
        except Exception as e:
            st.error(f"Error calculando frontera eficiente: {e}")
            return None
    
    @staticmethod
    def export_cart_to_excel():
        """Exportar carrito a Excel"""
        SimpleCart.initialize()
        
        if not st.session_state.selected_funds:
            return None
        
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Hoja 1: Composición del carrito
                cart_data = []
                for ticker in st.session_state.selected_funds:
                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                    cart_data.append({
                        'Ticker': ticker,
                        'Peso (%)': round(weight, 2)
                    })
                
                cart_df = pd.DataFrame(cart_data)
                cart_df.to_excel(writer, sheet_name='Composición Carrito', index=False)
                
                # Hoja 2: Información del carrito
                summary_data = {
                    'Métrica': [
                        'Número de Fondos',
                        'Peso Total (%)',
                        'Fecha de Creación'
                    ],
                    'Valor': [
                        len(st.session_state.selected_funds),
                        round(sum(st.session_state.portfolio_weights.values()), 2),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Información General', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error exportando carrito: {e}")
            return None
    
    @staticmethod
    def render_cart_analysis():
        """Renderizar análisis completo del carrito"""
        SimpleCart.initialize()
        
        st.markdown("# 📊 Análisis de Mi Carrito")
        
        # Botón para volver
        if st.button("← Volver al Dashboard Principal"):
            st.session_state.show_cart_analysis = False
            st.rerun()
        
        if not st.session_state.selected_funds:
            st.info("🛒 Tu carrito está vacío. Ve al dashboard principal para seleccionar fondos.")
            return
        
        # Cargar datos (necesitamos acceso a los datos)
        try:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, 'data')
            
            funds_path = os.path.join(data_dir, 'funds_prices.csv')
            funds_data = pd.read_csv(funds_path)
            
            dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
            etf_dict = pd.read_csv(dict_path)
        except Exception as e:
            st.error(f"Error cargando datos: {e}")
            return
        
        # Layout principal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## ⚖️ Gestión de Pesos")
            
            # Editor de pesos
            with st.form("cart_weights_form"):
                st.markdown("**Ajusta el peso de cada fondo en tu carrito:**")
                
                new_weights = {}
                
                for ticker in st.session_state.selected_funds:
                    current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                    
                    # Obtener información del fondo
                    fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                    fund_name = ticker  # Default
                    
                    if not fund_info.empty:
                        # Try different name columns
                        for name_col in ['Fund Name', 'Indice', 'Ticker']:
                            if name_col in fund_info.columns and pd.notna(fund_info[name_col].iloc[0]):
                                fund_name = fund_info[name_col].iloc[0]
                                break
                    
                    col_name, col_weight, col_remove = st.columns([3, 1.5, 1])
                    
                    with col_name:
                        st.markdown(f"**{fund_name}**")
                        st.caption(f"Ticker: {ticker}")
                        
                        # Mostrar clasificación del fondo
                        if not fund_info.empty:
                            classification_parts = []
                            if 'Asset Class' in fund_info.columns and pd.notna(fund_info['Asset Class'].iloc[0]):
                                classification_parts.append(f"🏛️ {fund_info['Asset Class'].iloc[0]}")
                            if 'Geografia' in fund_info.columns and pd.notna(fund_info['Geografia'].iloc[0]):
                                classification_parts.append(f"🌍 {fund_info['Geografia'].iloc[0]}")
                            if 'Sector' in fund_info.columns and pd.notna(fund_info['Sector'].iloc[0]):
                                classification_parts.append(f"🏭 {fund_info['Sector'].iloc[0]}")
                            
                            if classification_parts:
                                st.caption(" | ".join(classification_parts))
                    
                    with col_weight:
                        new_weights[ticker] = st.number_input(
                            "Peso %",
                            min_value=0.0,
                            max_value=100.0,
                            value=float(current_weight),
                            step=0.5,
                            key=f"cart_weight_{ticker}",
                            label_visibility="collapsed"
                        )
                    
                    with col_remove:
                        st.write("")  # Espaciado
                        if st.form_submit_button("🗑️", help="Remover del carrito"):
                            st.session_state.selected_funds.discard(ticker)
                            if ticker in st.session_state.portfolio_weights:
                                del st.session_state.portfolio_weights[ticker]
                            st.rerun()
                
                # Botones de acción
                st.markdown("---")
                col_update, col_normalize, col_equal = st.columns(3)
                
                with col_update:
                    if st.form_submit_button("💾 Actualizar Pesos", use_container_width=True):
                        for ticker, weight in new_weights.items():
                            st.session_state.portfolio_weights[ticker] = weight
                        st.success("✅ Pesos actualizados")
                        st.rerun()
                
                with col_normalize:
                    if st.form_submit_button("⚖️ Normalizar 100%", use_container_width=True):
                        total = sum(st.session_state.portfolio_weights.values())
                        if total > 0:
                            for ticker in st.session_state.portfolio_weights:
                                st.session_state.portfolio_weights[ticker] = (
                                    st.session_state.portfolio_weights[ticker] / total * 100
                                )
                        st.success("✅ Pesos normalizados")
                        st.rerun()
                
                with col_equal:
                    if st.form_submit_button("🔄 Pesos Iguales", use_container_width=True):
                        SimpleCart._rebalance_weights()
                        st.success("✅ Pesos distribuidos equitativamente")
                        st.rerun()
        
        with col2:
            st.markdown("## 📈 Resumen")
            
            # Métricas básicas
            num_funds = len(st.session_state.selected_funds)
            total_weight = sum(st.session_state.portfolio_weights.values())
            
            st.metric("Fondos Seleccionados", num_funds)
            st.metric("Peso Total", f"{total_weight:.1f}%")
            
            # Verificación de pesos
            if abs(total_weight - 100) > 0.1:
                if total_weight > 100:
                    st.warning(f"⚠️ Pesos suman {total_weight:.1f}%")
                else:
                    st.info(f"ℹ️ Pesos suman {total_weight:.1f}%")
            else:
                st.success(f"✅ Pesos balanceados")
            
            # Resumen por Asset Class
            st.markdown("### 🏛️ Por Asset Class")
            asset_class_weights = {}
            for ticker in st.session_state.selected_funds:
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                if not fund_info.empty and 'Asset Class' in fund_info.columns:
                    asset_class = fund_info['Asset Class'].iloc[0]
                    if pd.notna(asset_class):
                        weight = st.session_state.portfolio_weights.get(ticker, 0)
                        asset_class_weights[asset_class] = asset_class_weights.get(asset_class, 0) + weight
            
            for asset_class, weight in sorted(asset_class_weights.items()):
                st.caption(f"• {asset_class}: {weight:.1f}%")
            
            # Resumen por Geografía
            st.markdown("### 🌍 Por Geografía")
            geo_weights = {}
            for ticker in st.session_state.selected_funds:
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                if not fund_info.empty and 'Geografia' in fund_info.columns:
                    geografia = fund_info['Geografia'].iloc[0]
                    if pd.notna(geografia):
                        weight = st.session_state.portfolio_weights.get(ticker, 0)
                        geo_weights[geografia] = geo_weights.get(geografia, 0) + weight
            
            for geografia, weight in sorted(geo_weights.items()):
                st.caption(f"• {geografia}: {weight:.1f}%")
            
            # Lista de fondos
            st.markdown("### 📋 Fondos en Carrito")
            for ticker in st.session_state.selected_funds:
                weight = st.session_state.portfolio_weights.get(ticker, 0)
                st.caption(f"• {ticker}: {weight:.1f}%")
        
        # Análisis de fondos individuales
        st.markdown("## 📊 Análisis de Fondos Individuales")
        
        # Calcular métricas para cada fondo
        individual_metrics = []
        for ticker in st.session_state.selected_funds:
            metrics = SimpleCart.calculate_individual_fund_metrics(funds_data, ticker)
            if metrics:
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                
                # Obtener nombre del fondo
                fund_name = ticker
                if not fund_info.empty:
                    for name_col in ['Fund Name', 'Indice', 'Ticker']:
                        if name_col in fund_info.columns and pd.notna(fund_info[name_col].iloc[0]):
                            fund_name = fund_info[name_col].iloc[0]
                            break
                
                # Obtener información de clasificación
                asset_class = ""
                geografia = ""
                sector = ""
                
                if not fund_info.empty:
                    if 'Asset Class' in fund_info.columns and pd.notna(fund_info['Asset Class'].iloc[0]):
                        asset_class = fund_info['Asset Class'].iloc[0]
                    if 'Geografia' in fund_info.columns and pd.notna(fund_info['Geografia'].iloc[0]):
                        geografia = fund_info['Geografia'].iloc[0]
                    if 'Sector' in fund_info.columns and pd.notna(fund_info['Sector'].iloc[0]):
                        sector = fund_info['Sector'].iloc[0]
                
                row = {
                    'Ticker': ticker, 
                    'Fund Name': fund_name,
                    'Asset Class': asset_class,
                    'Geografia': geografia,
                    'Sector': sector
                }
                row.update(metrics)
                individual_metrics.append(row)
        
        if individual_metrics:
            df_individual = pd.DataFrame(individual_metrics)
            
            # Formatear columnas de porcentaje
            percentage_cols = ['YTD Return (%)', 'Monthly Return (%)', '1Y Return (%)', 
                              '2024 Return (%)', '2023 Return (%)', '2022 Return (%)',
                              'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']
            
            display_df = df_individual.copy()
            for col in percentage_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
        
        # Análisis de portafolio
        st.markdown("## 📈 Análisis de Portafolio")
        
        col_start, col_end, col_analyze = st.columns([2, 2, 1])
        
        with col_start:
            start_date = st.date_input(
                "📅 Fecha de Inicio",
                value=datetime.now().date() - timedelta(days=365),
                key="cart_analysis_start"
            )
        
        with col_end:
            end_date = st.date_input(
                "📅 Fecha de Fin",
                value=datetime.now().date(),
                key="cart_analysis_end"
            )
        
        with col_analyze:
            st.write("")  # Espaciado
            analyze_button = st.button("🔄 Analizar", use_container_width=True)
        
        if analyze_button:
            portfolio_analysis = SimpleCart.calculate_portfolio_metrics(
                funds_data, 
                pd.to_datetime(start_date), 
                pd.to_datetime(end_date)
            )
            
            if portfolio_analysis:
                # Mostrar métricas
                st.markdown("### 📊 Métricas del Portafolio")
                
                metrics = portfolio_analysis['metrics']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Retorno Total", f"{metrics['Retorno Total (%)']:.2f}%")
                    st.metric("Volatilidad", f"{metrics['Volatilidad Anualizada (%)']:.2f}%")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")
                    st.metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
                
                with col3:
                    st.metric("VaR 5%", f"{metrics['VaR 5% (%)']:.2f}%")
                    st.metric("CVaR 5%", f"{metrics['CVaR 5% (%)']:.2f}%")
                
                # Gráfico de performance
                st.markdown("### 📈 Performance Histórica")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=portfolio_analysis['chart_data']['dates'],
                    y=portfolio_analysis['chart_data']['values'],
                    mode='lines',
                    name='Mi Carrito',
                    line=dict(color='#60a5fa', width=3),
                    hovertemplate='<b>Mi Carrito</b><br>Fecha: %{x}<br>Valor: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Performance del Carrito (Base 100)",
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
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo calcular el análisis del portafolio. Verifica las fechas y los datos.")
        
        # Frontera eficiente
        if len(st.session_state.selected_funds) >= 2:
            st.markdown("## 🎯 Frontera Eficiente")
            
            if st.button("🔄 Calcular Frontera Eficiente"):
                with st.spinner("Calculando frontera eficiente..."):
                    efficient_frontier = SimpleCart.calculate_efficient_frontier(
                        funds_data,
                        list(st.session_state.selected_funds),
                        pd.to_datetime(start_date),
                        pd.to_datetime(end_date)
                    )
                    
                    if efficient_frontier:
                        # Gráfico de frontera eficiente
                        fig = go.Figure()
                        
                        # Frontera eficiente
                        risks = [p['risk'] for p in efficient_frontier['portfolios']]
                        returns = [p['return'] for p in efficient_frontier['portfolios']]
                        
                        fig.add_trace(go.Scatter(
                            x=risks,
                            y=returns,
                            mode='lines+markers',
                            name='Frontera Eficiente',
                            line=dict(color='#10b981', width=3),
                            marker=dict(size=4)
                        ))
                        
                        # Fondos individuales
                        fig.add_trace(go.Scatter(
                            x=efficient_frontier['risks'],
                            y=efficient_frontier['expected_returns'],
                            mode='markers',
                            name='Fondos Individuales',
                            marker=dict(size=10, color='#ef4444'),
                            text=efficient_frontier['assets'],
                            textposition='top center'
                        ))
                        
                        fig.update_layout(
                            title="Frontera Eficiente - Fondos Seleccionados",
                            xaxis_title="Riesgo (Volatilidad %)",
                            yaxis_title="Retorno Esperado (%)",
                            height=500,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#fafafa'),
                            xaxis=dict(gridcolor='#404040', color='#fafafa'),
                            yaxis=dict(gridcolor='#404040', color='#fafafa'),
                            title_font=dict(size=16, color='#fafafa')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("💡 La frontera eficiente muestra las mejores combinaciones de riesgo-retorno para tus fondos seleccionados.")
                    else:
                        st.error("No se pudo calcular la frontera eficiente. Se necesitan al menos 2 fondos con suficientes datos históricos.")
        else:
            st.info("💡 Selecciona al menos 2 fondos para calcular la frontera eficiente.")
        
        # Exportación
        st.markdown("## 💾 Exportar Carrito")
        
        excel_data = SimpleCart.export_cart_to_excel()
        if excel_data:
            st.download_button(
                label="📊 Descargar Carrito (Excel)",
                data=excel_data,
                file_name=f"mi_carrito_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

# Función para integrar el carrito simple
def integrate_simple_cart():
    """Función para integrar el carrito simple en cualquier dashboard"""
    SimpleCart.initialize()
    SimpleCart.render_cart_sidebar()
    
    # Si se debe mostrar el análisis del carrito
    if st.session_state.get('show_cart_analysis', False):
        SimpleCart.render_cart_analysis()
        return True  # Indica que se está mostrando el análisis del carrito
    
    return False  # Indica que se debe mostrar el contenido normal del dashboard