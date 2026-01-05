#!/usr/bin/env python3
"""
SIMPLE CART MODULE - FIXED VERSION
Módulo del carrito simple con checkboxes - versión corregida sin errores de formularios
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import cvxpy as cp
import io
from datetime import datetime, timedelta

# SHARED METRICS CALCULATOR (Sincronizado en ambos dashboards)
from metrics_calculator import calculate_individual_fund_metrics, calculate_portfolio_metrics

class PortfolioManager:
    """Clase para manejar la gestión de portafolios"""
    
    @staticmethod
    def initialize():
        """Inicializar el estado del portafolio"""
        if 'selected_funds' not in st.session_state:
            st.session_state.selected_funds = set()
        if 'portfolio_weights' not in st.session_state:
            st.session_state.portfolio_weights = {}
        if 'show_portfolio_analysis' not in st.session_state:
            st.session_state.show_portfolio_analysis = False
            
    @staticmethod
    def render_fund_selector(ticker, fund_name, key_suffix=""):
        """Renderizar selector para fondos del portafolio"""
        PortfolioManager.initialize()
        
        # Toggle button para selección con un solo click
        is_selected = ticker in st.session_state.selected_funds
        
        if st.button(
            f"{'✓ ' if is_selected else '+ '}{fund_name}",
            key=f"select_{ticker}_{key_suffix}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            # Toggle: agregar o remover del portafolio
            if ticker in st.session_state.selected_funds:
                # Remover del portafolio
                st.session_state.selected_funds.discard(ticker)
                if ticker in st.session_state.portfolio_weights:
                    del st.session_state.portfolio_weights[ticker]
                
                # Rebalancear fondos restantes
                remaining_funds = len(st.session_state.selected_funds)
                if remaining_funds > 0:
                    equal_weight = 100.0 / remaining_funds
                    for remaining_ticker in st.session_state.selected_funds:
                        st.session_state.portfolio_weights[remaining_ticker] = equal_weight
            else:
                # Agregar al portafolio
                st.session_state.selected_funds.add(ticker)
                if ticker not in st.session_state.portfolio_weights:
                    # Peso inicial equitativo
                    num_funds = len(st.session_state.selected_funds)
                    equal_weight = 100.0 / num_funds if num_funds > 0 else 0
                    st.session_state.portfolio_weights[ticker] = equal_weight
                    
                    # Rebalancear otros fondos
                    for other_ticker in st.session_state.selected_funds:
                        st.session_state.portfolio_weights[other_ticker] = equal_weight
            
            st.rerun()
    
    @staticmethod
    def render_portfolio_sidebar():
        """Renderizar widget del portafolio en el sidebar"""
        PortfolioManager.initialize()
        
        st.sidebar.markdown("## Portfolio Manager")
        
        num_funds = len(st.session_state.selected_funds)
        
        if num_funds == 0:
            st.sidebar.info("No assets selected")
            return
        
        st.sidebar.metric("Selected Assets", num_funds)
        
        # Lista de fondos con pesos
        st.sidebar.markdown("### Asset Allocation:")
        for ticker in st.session_state.selected_funds:
            weight = st.session_state.portfolio_weights.get(ticker, 0)
            st.sidebar.caption(f"• {ticker}: {weight:.1f}%")
        
        # Verificar total de pesos
        total_weight = sum(st.session_state.portfolio_weights.values())
        if abs(total_weight - 100) > 0.1:
            st.sidebar.warning(f"Total: {total_weight:.1f}%")
        else:
            st.sidebar.success("Total: 100%")
        
        # Botones de acción
        if st.sidebar.button("Analyze Portfolio", use_container_width=True):
            st.session_state.show_portfolio_analysis = True
            st.rerun()
        
        if st.sidebar.button("Clear Portfolio", use_container_width=True):
            st.session_state.selected_funds.clear()
            st.session_state.portfolio_weights.clear()
            st.rerun()
    
    @staticmethod
    def calculate_individual_fund_metrics(funds_data, ticker):
        """
        ✅ VERSIÓN SINCRONIZADA - Wrapper que usa metrics_calculator.py
        Garantiza consistencia entre dashboard principal y análisis de portafolio
        """
        return calculate_individual_fund_metrics(funds_data, ticker)
    
    @staticmethod
    def calculate_portfolio_metrics(funds_data, selected_funds, weights, start_date=None, end_date=None, returns_data=None):
        """
        ✅ VERSIÓN SINCRONIZADA - Wrapper que usa metrics_calculator.py
        Garantiza consistencia entre dashboard principal y análisis de portafolio
        """
        return calculate_portfolio_metrics(funds_data, selected_funds, weights, start_date, end_date, returns_data)
    
    @staticmethod
    def calculate_efficient_frontier(funds_data, selected_funds, start_date=None, end_date=None):
        """Calcular frontera eficiente OPTIMIZADA para fondos del carrito"""
        try:
            funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
            
            if len(selected_funds) < 2:
                return None
            
            # OPTIMIZACIÓN 1: Usar solo los fondos seleccionados y período común
            relevant_columns = ['Dates'] + [fund for fund in selected_funds if fund in funds_data.columns]
            filtered_data = funds_data[relevant_columns].dropna()
            
            if len(filtered_data) < 50:  # Mínimo 50 observaciones para estabilidad
                st.warning(f"⚠️ Datos insuficientes: {len(filtered_data)} observaciones (mínimo 50)")
                return None
            
            # OPTIMIZACIÓN 2: Calcular retornos una sola vez
            returns_df = filtered_data.set_index('Dates').pct_change().dropna()
            
            if len(returns_df) < 30:
                st.warning(f"⚠️ Retornos insuficientes: {len(returns_df)} observaciones")
                return None
            
            st.success(f"✅ **Usando {len(returns_df)} observaciones de retornos diarios**")
            
            # OPTIMIZACIÓN 3: Cálculos vectorizados
            expected_returns = returns_df.mean() * 252  # Anualizados
            cov_matrix = returns_df.cov() * 252  # Anualizada
            
            # OPTIMIZACIÓN 4: Menos puntos en la frontera (25 en lugar de 50)
            n_points = 25
            target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
            
            efficient_portfolios = []
            n_assets = len(expected_returns)
            
            # OPTIMIZACIÓN 5: Solver más rápido y configuración optimizada
            for target in target_returns:
                try:
                    weights = cp.Variable(n_assets)
                    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
                    
                    constraints = [
                        cp.sum(weights) == 1,
                        weights >= 0,
                        expected_returns.values @ weights == target
                    ]
                    
                    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                    # Usar OSQP que es más rápido para problemas cuadráticos
                    problem.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-4, eps_rel=1e-4)
                    
                    if problem.status == cp.OPTIMAL:
                        portfolio_return = target
                        portfolio_risk = np.sqrt(problem.value)
                        efficient_portfolios.append({
                            'return': portfolio_return * 100,  # Ya anualizado
                            'risk': portfolio_risk * 100,     # Ya anualizado
                            'weights': weights.value
                        })
                
                except:
                    continue
            
            # OPTIMIZACIÓN 6: Retornar datos ya procesados
            individual_risks = np.sqrt(np.diag(cov_matrix)) * 100  # Ya anualizados
            individual_returns = expected_returns * 100  # Ya anualizados
            
            return {
                'portfolios': efficient_portfolios,
                'assets': list(expected_returns.index),
                'expected_returns': individual_returns,
                'risks': individual_risks,
                'returns_data': returns_df  # Para cálculos adicionales
            }
            
        except Exception as e:
            st.error(f"Error calculando frontera eficiente: {e}")
            return None
    
    @staticmethod
    def export_portfolio_to_excel():
        """Exportar portafolio a Excel"""
        try:
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Composición del carrito
                composition_data = []
                for ticker in st.session_state.selected_funds:
                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                    composition_data.append({
                        'Ticker': ticker,
                        'Peso (%)': weight
                    })
                
                composition_df = pd.DataFrame(composition_data)
                composition_df.to_excel(writer, sheet_name='Composición', index=False)
                
                # Información general
                summary_data = {
                    'Fecha de Exportación': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Número de Fondos': [len(st.session_state.selected_funds)],
                    'Peso Total': [f"{sum(st.session_state.portfolio_weights.values()):.1f}%"]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Información General', index=False)
            
            output.seek(0)
            return output.getvalue()
            
        except Exception as e:
            st.error(f"Error exportando carrito: {e}")
            return None
    
    @staticmethod
    def render_portfolio_analysis():
        """Renderizar análisis completo del portafolio"""
        PortfolioManager.initialize()
        
        # Botón para volver al dashboard principal
        if st.button("← Back to Main Dashboard"):
            st.session_state.show_portfolio_analysis = False
            st.rerun()
        
        st.markdown("# Portfolio Analysis")
        
        if not st.session_state.selected_funds:
            st.info("No assets selected. Go to the main dashboard to select funds.")
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
        
        # Layout en dos columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## Asset Allocation Management")
            
            # Editor de pesos (SIN FORMULARIO para evitar problemas)
            st.markdown("**Adjust the weight of each fund in your portfolio:**")
            
            funds_to_remove = []
            
            for ticker in st.session_state.selected_funds:
                current_weight = st.session_state.portfolio_weights.get(ticker, 0)
                
                # Obtener información del fondo
                fund_info = etf_dict[etf_dict['Ticker'] == ticker]
                fund_name = ticker  # Default
                
                if not fund_info.empty:
                    # Prioritize "Indice" for more intuitive fund names
                    if 'Indice' in fund_info.columns and pd.notna(fund_info['Indice'].iloc[0]):
                        fund_name = fund_info['Indice'].iloc[0]
                    elif 'Fund Name' in fund_info.columns and pd.notna(fund_info['Fund Name'].iloc[0]):
                        fund_name = fund_info['Fund Name'].iloc[0]
                
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
                    new_weight = st.number_input(
                        "Peso %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(current_weight),
                        step=0.5,
                        key=f"cart_weight_{ticker}",
                        label_visibility="collapsed"
                    )
                    # Actualizar automáticamente
                    st.session_state.portfolio_weights[ticker] = new_weight
                
                with col_remove:
                    st.write("")  # Espaciado
                    if st.button("🗑️", key=f"remove_{ticker}", help="Remove from portfolio"):
                        funds_to_remove.append(ticker)
            
            # Procesar eliminaciones
            for ticker in funds_to_remove:
                st.session_state.selected_funds.discard(ticker)
                if ticker in st.session_state.portfolio_weights:
                    del st.session_state.portfolio_weights[ticker]
                st.rerun()
            
            # Botones de acción
            st.markdown("---")
            col_normalize, col_equal = st.columns(2)
            
            with col_normalize:
                if st.button("⚖️ Normalizar 100%", use_container_width=True):
                    total = sum(st.session_state.portfolio_weights.values())
                    if total > 0:
                        for ticker in st.session_state.portfolio_weights:
                            st.session_state.portfolio_weights[ticker] = (st.session_state.portfolio_weights[ticker] / total) * 100
                    st.success("✅ Pesos normalizados a 100%")
                    st.rerun()
            
            with col_equal:
                if st.button("🟰 Pesos Iguales", use_container_width=True):
                    equal_weight = 100.0 / len(st.session_state.selected_funds)
                    for ticker in st.session_state.selected_funds:
                        st.session_state.portfolio_weights[ticker] = equal_weight
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
            metrics = PortfolioManager.calculate_individual_fund_metrics(funds_data, ticker)
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
            
            # Format percentage columns + Sharpe
            percentage_cols = ['YTD Return (%)', 'Monthly Return (%)',
                               '2025 Return (%)', '2024 Return (%)', '2023 Return (%)',
                              'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)']

            display_df = df_individual.copy()
            for col in percentage_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            if 'Sharpe Ratio' in display_df.columns:
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")
            
            st.dataframe(display_df, use_container_width=True)
        
        # Análisis de portafolio
        st.markdown("## 📈 Análisis de Portafolio")
        
        # Analizar fechas de inception de los fondos seleccionados
        funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
        
        # Información de inception por fondo
        st.markdown("### 📅 Fechas de Inception de Activos")
        inception_info = []
        portfolio_min_date = None
        
        for fund in st.session_state.selected_funds:
            if fund in funds_data.columns:
                fund_data = funds_data[['Dates', fund]].dropna()
                if len(fund_data) > 0:
                    inception_date = fund_data['Dates'].min()
                    last_date = fund_data['Dates'].max()
                    inception_info.append({
                        'Asset': fund,
                        'Inception Date': inception_date.strftime('%Y-%m-%d'),
                        'Last Date': last_date.strftime('%Y-%m-%d'),
                        'Days Available': len(fund_data)
                    })
                    
                    # Encontrar la fecha más tardía de inception (cuando todos los fondos están disponibles)
                    if portfolio_min_date is None or inception_date > portfolio_min_date:
                        portfolio_min_date = inception_date
        
        if inception_info:
            inception_df = pd.DataFrame(inception_info)
            st.dataframe(inception_df, use_container_width=True, hide_index=True)
            
            if portfolio_min_date:
                st.info(f"📊 **Portfolio completo disponible desde**: {portfolio_min_date.strftime('%Y-%m-%d')} (cuando todos los activos tienen datos)")
        
        # Date range selector con límites inteligentes
        min_date = funds_data['Dates'].min().date()
        max_date = funds_data['Dates'].max().date()
        
        # Sugerir fecha de inicio basada en disponibilidad del portafolio
        suggested_start = portfolio_min_date.date() if portfolio_min_date else min_date
        default_start = max(suggested_start, max_date - pd.Timedelta(days=365).to_pytimedelta())
        
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input(
                "Fecha de inicio:", 
                value=default_start, 
                min_value=min_date, 
                max_value=max_date,
                help=f"Portfolio completo disponible desde {suggested_start}"
            )
        with col_end:
            end_date = st.date_input("Fecha de fin:", value=max_date, min_value=min_date, max_value=max_date)
        
        if start_date < end_date:
            # Verificar si todas las fechas están disponibles
            missing_funds = []
            for fund in st.session_state.selected_funds:
                if fund in funds_data.columns:
                    fund_data = funds_data[['Dates', fund]].dropna()
                    fund_start = fund_data['Dates'].min().date()
                    if fund_start > start_date:
                        missing_funds.append(f"{fund} (disponible desde {fund_start})")
            
            if missing_funds:
                st.warning(f"⚠️ **Fondos con datos faltantes en el período seleccionado:**\n" + "\n".join([f"• {fund}" for fund in missing_funds]))
            
            # Calculate portfolio metrics con fechas específicas
            portfolio_metrics = PortfolioManager.calculate_portfolio_metrics(
                funds_data, 
                st.session_state.selected_funds, 
                st.session_state.portfolio_weights,
                start_date=pd.to_datetime(start_date),
                end_date=pd.to_datetime(end_date)
            )
            
            if portfolio_metrics:
                # Mostrar información del período analizado
                st.success(f"📊 **Período analizado**: {portfolio_metrics['start_date'].strftime('%Y-%m-%d')} a {portfolio_metrics['end_date'].strftime('%Y-%m-%d')} ({portfolio_metrics['period_days']} días de historia)")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Retorno Total", f"{portfolio_metrics['total_return']:.2f}%")
                with col2:
                    st.metric("Retorno Anualizado", f"{portfolio_metrics['annualized_return']:.2f}%")
                with col3:
                    st.metric("Volatilidad", f"{portfolio_metrics['volatility']:.2f}%")
                with col4:
                    st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.3f}")
                
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2f}%")
                with col6:
                    st.metric("VaR 5%", f"{portfolio_metrics['var_5']:.2f}%")
                with col7:
                    st.metric("CVaR 5%", f"{portfolio_metrics['cvar_5']:.2f}%")
                
                # Performance chart
                st.markdown("### 📊 Evolución del Portafolio")
                
                # Create cumulative performance chart with dates on x-axis
                cumulative_returns = (1 + portfolio_metrics['portfolio_returns']).cumprod() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name='Portafolio',
                    line=dict(color='#10b981', width=2),
                    hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Valor: %{y:.2f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Evolución del Portafolio (Base 100)",
                    xaxis_title="Fecha",
                    yaxis_title="Valor (Base 100)",
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa'),
                    xaxis=dict(gridcolor='#404040', color='#fafafa'),
                    yaxis=dict(gridcolor='#404040', color='#fafafa')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Frontera eficiente del carrito
        if len(st.session_state.selected_funds) >= 2:
            st.markdown("## Portfolio Efficient Frontier")
            
            if st.button("Calculate Efficient Frontier"):
                with st.spinner("Calculating efficient frontier..."):
                    efficient_frontier = PortfolioManager.calculate_efficient_frontier(
                        funds_data, 
                        list(st.session_state.selected_funds), 
                        pd.to_datetime(start_date), 
                        pd.to_datetime(end_date)
                    )
                    
                    if efficient_frontier:
                        # Create efficient frontier chart (ESTILO ORIGINAL)
                        fig = go.Figure()
                        
                        # Efficient frontier line
                        risks = [p['risk'] for p in efficient_frontier['portfolios']]
                        returns = [p['return'] for p in efficient_frontier['portfolios']]
                        
                        fig.add_trace(go.Scatter(
                            x=risks,
                            y=returns,
                            mode='lines+markers',
                            name='Efficient Frontier',
                            line=dict(color='#60a5fa', width=3),
                            marker=dict(size=4),
                            hovertemplate='Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Individual assets
                        fig.add_trace(go.Scatter(
                            x=efficient_frontier['risks'],
                            y=efficient_frontier['expected_returns'],
                            mode='markers+text',
                            name='Individual Assets',
                            marker=dict(size=12, color='#9ca3af', symbol='diamond'),
                            text=efficient_frontier['assets'],
                            textposition='top center',
                            hovertemplate='Asset: %{text}<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
                        ))
                        
                        # Calculate current portfolio metrics using same data as efficient frontier
                        portfolio_metrics = PortfolioManager.calculate_portfolio_metrics(
                            funds_data, 
                            list(st.session_state.selected_funds), 
                            st.session_state.portfolio_weights,
                            returns_data=efficient_frontier['returns_data']  # Usar los mismos datos
                        )
                        
                        if portfolio_metrics:
                            # Add current portfolio point
                            fig.add_trace(go.Scatter(
                                x=[portfolio_metrics['volatility']],
                                y=[portfolio_metrics['annualized_return']],
                                mode='markers+text',
                                name='Current Portfolio',
                                marker=dict(size=20, color='#34d399', symbol='star', line=dict(width=2, color='#ffffff')),
                                text=['Your Portfolio'],
                                textposition='top center',
                                textfont=dict(size=12, color='#ffffff'),
                                hovertemplate='Current Portfolio<br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: ' + f'{portfolio_metrics["sharpe_ratio"]:.3f}' + '<extra></extra>'
                            ))
                        
                        fig.update_layout(
                            title="Efficient Frontier - Risk vs Return",
                            xaxis_title="Risk (Volatility %)",
                            yaxis_title="Expected Return (%)",
                            height=600,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#fafafa'),
                            xaxis=dict(gridcolor='#404040', color='#fafafa'),
                            yaxis=dict(gridcolor='#404040', color='#fafafa'),
                            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#404040', borderwidth=1),
                            title_font=dict(size=18, color='#fafafa')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Portfolio Analysis vs Efficient Frontier
                        if portfolio_metrics:
                            st.markdown("### Portfolio Analysis")
                            
                            # Show current portfolio composition
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("#### Current Portfolio Composition")
                                composition_data = []
                                for ticker in st.session_state.selected_funds:
                                    weight = st.session_state.portfolio_weights.get(ticker, 0)
                                    composition_data.append({
                                        'Asset': ticker,
                                        'Weight (%)': f"{weight:.1f}%"
                                    })
                                
                                composition_df = pd.DataFrame(composition_data)
                                st.dataframe(composition_df, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.markdown("#### Portfolio Metrics (Annualized)")
                                col2a, col2b = st.columns(2)
                                with col2a:
                                    st.metric("Return", f"{portfolio_metrics['annualized_return']:.2f}%")
                                    st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.3f}")
                                with col2b:
                                    st.metric("Risk", f"{portfolio_metrics['volatility']:.2f}%")
                                    st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2f}%")
                            
                            # Find closest point on efficient frontier
                            min_distance = float('inf')
                            closest_portfolio = None
                            
                            for portfolio in efficient_frontier['portfolios']:
                                # Calculate Euclidean distance in risk-return space
                                distance = np.sqrt(
                                    (portfolio['risk'] - portfolio_metrics['volatility'])**2 + 
                                    (portfolio['return'] - portfolio_metrics['annualized_return'])**2
                                )
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_portfolio = portfolio
                            
                            if closest_portfolio:
                                st.markdown("#### Efficiency Analysis")
                                
                                # Calculate efficiency metrics
                                return_diff = portfolio_metrics['annualized_return'] - closest_portfolio['return']
                                risk_diff = portfolio_metrics['volatility'] - closest_portfolio['risk']
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    if return_diff >= 0:
                                        st.success(f"✅ Return advantage: +{return_diff:.2f}% vs efficient frontier")
                                    else:
                                        st.warning(f"⚠️ Return gap: {return_diff:.2f}% vs efficient frontier")
                                
                                with col2:
                                    if risk_diff <= 0:
                                        st.success(f"✅ Risk advantage: {risk_diff:.2f}% vs efficient frontier")
                                    else:
                                        st.warning(f"⚠️ Extra risk: +{risk_diff:.2f}% vs efficient frontier")
                                
                                # Overall efficiency assessment
                                if return_diff >= -0.5 and risk_diff <= 1.0:
                                    st.success("🎯 **Your portfolio is well-aligned with the efficient frontier!**")
                                elif return_diff >= -1.0 and risk_diff <= 2.0:
                                    st.info("📊 **Your portfolio is reasonably efficient with room for improvement.**")
                                else:
                                    st.warning("⚡ **Consider rebalancing to improve efficiency.**")
                        
                        # Show optimal portfolios
                        st.markdown("### Optimal Portfolios")
                        
                        # Find portfolios with highest Sharpe ratio
                        sharpe_ratios = []
                        for portfolio in efficient_frontier['portfolios']:
                            if portfolio['risk'] > 0:
                                sharpe = portfolio['return'] / portfolio['risk']
                                sharpe_ratios.append(sharpe)
                            else:
                                sharpe_ratios.append(0)
                        
                        if sharpe_ratios:
                            best_sharpe_idx = np.argmax(sharpe_ratios)
                            best_portfolio = efficient_frontier['portfolios'][best_sharpe_idx]
                            
                            st.markdown("**Portfolio with Highest Sharpe Ratio:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Expected Return", f"{best_portfolio['return']:.2f}%")
                            with col2:
                                st.metric("Risk (Volatility)", f"{best_portfolio['risk']:.2f}%")
                            with col3:
                                st.metric("Sharpe Ratio", f"{sharpe_ratios[best_sharpe_idx]:.3f}")
                            
                            # Show weights
                            weights_df = pd.DataFrame({
                                'Asset': efficient_frontier['assets'],
                                'Weight (%)': [w * 100 for w in best_portfolio['weights']]
                            })
                            weights_df = weights_df[weights_df['Weight (%)'] > 0.1]  # Show only significant weights
                            weights_df = weights_df.sort_values('Weight (%)', ascending=False)
                            
                            st.dataframe(weights_df, use_container_width=True)
                        
                        st.info("💡 The efficient frontier shows optimal risk-return combinations. Points on the frontier represent portfolios that maximize return for a given level of risk. Your portfolio (green star) shows how well-aligned your allocation is with optimal efficiency.")
                    
                    else:
                        st.error("No se pudo calcular la frontera eficiente. Asegúrate de tener al menos 2 fondos con datos históricos suficientes.")
        
        # Exportación
        st.markdown("## Export Portfolio")
        
        if st.button("Export to Excel", use_container_width=True):
            excel_data = PortfolioManager.export_portfolio_to_excel()
            if excel_data:
                st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

def integrate_portfolio_manager():
    """Función principal de integración del gestor de portafolio"""
    PortfolioManager.initialize()
    
    # Renderizar portafolio en sidebar
    PortfolioManager.render_portfolio_sidebar()
    
    # Si se debe mostrar el análisis del portafolio, renderizarlo
    if st.session_state.show_portfolio_analysis:
        PortfolioManager.render_portfolio_analysis()
        return True  # Indica que se está mostrando el análisis del portafolio
    
    return False  # Indica que se debe mostrar el dashboard principal