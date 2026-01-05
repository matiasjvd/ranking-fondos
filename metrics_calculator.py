#!/usr/bin/env python3
"""
SHARED METRICS CALCULATOR MODULE
Módulo centralizado para cálculo consistente de métricas
Usado por funds_dashboard.py y simple_cart_fixed.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def calculate_individual_fund_metrics(funds_data, fund_ticker):
    """
    Calculate comprehensive performance metrics for a fund.
    ✅ VERSIÓN OFICIAL - Usada en ambos dashboards
    
    Incluye:
    - Manejo robusto de datos faltantes (forward fill)
    - Limpieza de outliers (retornos > 50%)
    - Cálculo correcto de Sharpe anualizado
    """
    try:
        if fund_ticker not in funds_data.columns:
            return None
        
        # Get all data for this fund
        prices = funds_data[['Dates', fund_ticker]].copy()
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        
        # Find the first valid price (fund inception date)
        first_valid_idx = prices[fund_ticker].first_valid_index()
        if first_valid_idx is None:
            return None
        
        # Only use data from inception date onwards
        prices = prices.iloc[first_valid_idx:].reset_index(drop=True)
        
        # Forward fill prices to handle missing data after inception
        prices[fund_ticker] = prices[fund_ticker].ffill()
        
        # Remove any remaining NaN (shouldn't happen after ffill from inception)
        prices = prices.dropna()
        
        if len(prices) < 2:
            return None
        
        prices['Returns'] = prices[fund_ticker].pct_change()
        
        # Clean outliers: detect and remove extreme price jumps (likely data errors)
        # Use a threshold of 50% daily return as indicator of bad data
        # This catches decimal point errors and other data quality issues
        returns_clean = prices['Returns'].copy()
        outlier_threshold = 0.50  # 50% daily return threshold
        
        # Identify outliers
        outliers_mask = abs(returns_clean) > outlier_threshold
        
        if outliers_mask.any():
            # For rows with outlier returns, interpolate the price from neighbors
            outlier_indices = prices[outliers_mask].index.tolist()
            
            for idx in outlier_indices:
                if idx > 0 and idx < len(prices) - 1:
                    # Interpolate price from previous and next valid prices
                    prev_price = prices[fund_ticker].iloc[idx - 1]
                    next_idx = idx + 1
                    # Find next non-outlier
                    while next_idx < len(prices) and next_idx in outlier_indices:
                        next_idx += 1
                    
                    if next_idx < len(prices):
                        next_price = prices[fund_ticker].iloc[next_idx]
                        # Simple average interpolation
                        prices.loc[idx, fund_ticker] = (prev_price + next_price) / 2
                    else:
                        # Use previous price if no next price available
                        prices.loc[idx, fund_ticker] = prev_price
                elif idx == 0 and len(prices) > 1:
                    # First row: use next price
                    prices.loc[idx, fund_ticker] = prices[fund_ticker].iloc[1]
                elif idx == len(prices) - 1 and len(prices) > 1:
                    # Last row: use previous price
                    prices.loc[idx, fund_ticker] = prices[fund_ticker].iloc[-2]
            
            # Recalculate returns after cleaning
            prices['Returns'] = prices[fund_ticker].pct_change()
        
        current_date = prices['Dates'].max()
        current_year = current_date.year
        
        # YTD Return
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        ytd_return = ((ytd_data[fund_ticker].iloc[-1] / ytd_data[fund_ticker].iloc[0]) - 1) * 100 if len(ytd_data) > 1 else 0
        
        # Monthly Return (last 30 days)
        month_start = current_date - timedelta(days=30)
        month_data = prices[prices['Dates'] >= month_start]
        monthly_return = ((month_data[fund_ticker].iloc[-1] / month_data[fund_ticker].iloc[0]) - 1) * 100 if len(month_data) > 1 else 0
        
        # 1 Year Return
        year_1_start = current_date - timedelta(days=365)
        year_1_data = prices[prices['Dates'] >= year_1_start]
        return_1y = ((year_1_data[fund_ticker].iloc[-1] / year_1_data[fund_ticker].iloc[0]) - 1) * 100 if len(year_1_data) > 1 else 0
        
        # Annual returns for specific years
        returns_by_year = {}
        for year in [2025, 2024, 2023]:
            year_start = pd.to_datetime(f'{year}-01-01')
            year_end = pd.to_datetime(f'{year}-12-31')
            year_data = prices[(prices['Dates'] >= year_start) & (prices['Dates'] <= year_end)]
            if len(year_data) > 1:
                year_return = ((year_data[fund_ticker].iloc[-1] / year_data[fund_ticker].iloc[0]) - 1) * 100
                returns_by_year[f'{year} Return (%)'] = year_return
        
        # Volatility (annualized)
        volatility = prices['Returns'].std() * np.sqrt(252) * 100
        
        # Max Drawdown
        cumulative = (1 + prices['Returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # VaR and CVaR (5% confidence level, annualized) + Sharpe
        returns_clean = prices['Returns'].dropna()
        if len(returns_clean) > 0:
            # Standard VaR/CVaR calculation using all returns
            # This is correct for all fund types, including low-liquidity funds
            daily_var_5 = np.percentile(returns_clean, 5)
            var_5 = daily_var_5 * np.sqrt(252) * 100
            
            threshold = np.percentile(returns_clean, 5)
            worst_returns = returns_clean[returns_clean <= threshold]
            if len(worst_returns) > 0:
                daily_cvar_5 = worst_returns.mean()
                cvar_5 = daily_cvar_5 * np.sqrt(252) * 100
            else:
                cvar_5 = var_5
            
            # Annualized return and Sharpe Ratio (risk-free ~ 0)
            # Using geometric annualization for accuracy
            total_ret = (1 + returns_clean).prod() - 1
            ann_return = ((1 + total_ret) ** (252 / len(returns_clean))) - 1
            vol_ann = returns_clean.std() * np.sqrt(252)
            sharpe_ratio = ann_return / vol_ann if vol_ann > 0 else 0
        else:
            var_5 = 0
            cvar_5 = 0
            sharpe_ratio = 0
        
        metrics = {
            'YTD Return (%)': ytd_return,
            'Monthly Return (%)': monthly_return,
            'Volatility (%)': volatility,
            'Max Drawdown (%)': max_drawdown,
            'VaR 5% (%)': var_5,
            'CVaR 5% (%)': cvar_5,
            'Sharpe Ratio': sharpe_ratio
        }
        
        metrics.update(returns_by_year)
        
        return metrics
        
    except Exception as e:
        return None


def calculate_portfolio_metrics(funds_data, selected_funds, weights, start_date=None, end_date=None, returns_data=None):
    """
    Calculate portfolio metrics using individual fund metrics.
    ✅ VERSIÓN OFICIAL - Sincronizada con calculate_individual_fund_metrics
    
    Asegura que:
    - Usa los mismos datos limpios y preparados
    - Cálculos consistentes con fondos individuales
    """
    try:
        # Asegurarse que selected_funds contiene solo fondos válidos
        selected_funds = [f for f in selected_funds if f in funds_data.columns]
        
        if len(selected_funds) == 0:
            return None
        
        # OPTIMIZACIÓN: Si ya tenemos los datos de retornos, usarlos directamente
        if returns_data is not None:
            returns_df = returns_data
            # Si se especifican fechas, filtrar los datos de retornos
            if start_date is not None and end_date is not None:
                returns_df = returns_df[(returns_df.index >= start_date) & (returns_df.index <= end_date)]
        else:
            # Calcular retornos usando los mismos datos limpios que para fondos individuales
            funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
            
            # Preparar datos: forward fill y limpieza
            data_copy = funds_data[['Dates'] + selected_funds].copy()
            data_copy = data_copy.sort_values('Dates').reset_index(drop=True)
            
            # Forward fill para manejo consistente de datos faltantes
            for fund in selected_funds:
                first_valid_idx = data_copy[fund].first_valid_index()
                if first_valid_idx is not None:
                    data_copy[fund] = data_copy[fund].iloc[first_valid_idx:].ffill()
            
            # Eliminar NaNs
            data_copy = data_copy.dropna()
            
            if len(data_copy) < 2:
                return None
            
            # Filtrar por fechas si se proporcionan
            if start_date is not None and end_date is not None:
                data_copy = data_copy[(data_copy['Dates'] >= start_date) & (data_copy['Dates'] <= end_date)].copy()
            
            if len(data_copy) < 2:
                return None
            
            # Calcular retornos
            returns_df = data_copy.set_index('Dates')[selected_funds].pct_change().dropna()
        
        if len(returns_df) < 2:
            return None
        
        # Cálculo vectorizado de retornos del portafolio
        weights_array = np.array([weights.get(fund, 0) / 100 for fund in returns_df.columns])
        portfolio_returns = returns_df.dot(weights_array)
        
        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        
        # Annualized return usando la misma fórmula que en calculate_individual_fund_metrics
        annualized_return = ((1 + total_return) ** (252 / len(portfolio_returns))) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Max Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_5 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        cvar_5 = portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean() * np.sqrt(252)
        
        return {
            'total_return': total_return * 100,
            'annualized_return': annualized_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'var_5': var_5 * 100,
            'cvar_5': cvar_5 * 100,
            'portfolio_returns': portfolio_returns,
            'period_days': len(portfolio_returns),
            'start_date': portfolio_returns.index.min() if len(portfolio_returns) > 0 else None,
            'end_date': portfolio_returns.index.max() if len(portfolio_returns) > 0 else None
        }
        
    except Exception as e:
        return None