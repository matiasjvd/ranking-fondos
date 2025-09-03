#!/usr/bin/env python3
"""
FUNDS ANALYSIS DASHBOARD
Professional fund analysis with interactive filtering and performance metrics
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import os

import cvxpy as cp
import base64
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dashboard de Análisis de Fondos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    .performance-table {
        font-size: 0.9rem;
    }
    .positive-return {
        color: #059669;
        font-weight: 500;
    }
    .negative-return {
        color: #dc2626;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load funds data and dictionary from CSV files"""
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Load price data
        funds_path = os.path.join(data_dir, 'funds_prices.csv')
        funds = pd.read_csv(funds_path)
        
        # Load ETF dictionary
        dict_path = os.path.join(data_dir, 'funds_dictionary.csv')
        etf_dict = pd.read_csv(dict_path)
        
        return funds, etf_dict
        
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.error("Please check that the following files exist:")
        st.error(f"- {os.path.join(script_dir, 'data', 'funds_prices.csv')}")
        st.error(f"- {os.path.join(script_dir, 'data', 'funds_dictionary.csv')}")
        st.error("Run 'python convert_data.py' first to generate the CSV files.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def calculate_performance_metrics(funds_df, fund_ticker):
    """Calculate comprehensive performance metrics for a fund"""
    try:
        if fund_ticker not in funds_df.columns:
            return None
        
        # Get fund prices and calculate returns
        prices = funds_df[['Dates', fund_ticker]].dropna()
        if len(prices) < 2:
            return None
        
        prices['Dates'] = pd.to_datetime(prices['Dates'])
        prices = prices.sort_values('Dates').reset_index(drop=True)
        
        # Calculate returns
        prices['Returns'] = prices[fund_ticker].pct_change()
        
        # Current date for calculations
        current_date = prices['Dates'].max()
        current_year = current_date.year
        
        # YTD Return
        ytd_start = pd.to_datetime(f'{current_year}-01-01')
        ytd_data = prices[prices['Dates'] >= ytd_start]
        if len(ytd_data) > 1:
            ytd_return = ((ytd_data[fund_ticker].iloc[-1] / ytd_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            ytd_return = 0
        
        # MTD Return (Month-to-Date)
        mtd_start = pd.to_datetime(f'{current_date.year}-{current_date.month:02d}-01')
        mtd_data = prices[prices['Dates'] >= mtd_start]
        if len(mtd_data) > 1:
            mtd_return = ((mtd_data[fund_ticker].iloc[-1] / mtd_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            mtd_return = 0
        
        # Monthly Return (last 30 days)
        monthly_start = current_date - timedelta(days=30)
        monthly_data = prices[prices['Dates'] >= monthly_start]
        if len(monthly_data) > 1:
            monthly_return = ((monthly_data[fund_ticker].iloc[-1] / monthly_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            monthly_return = 0
        
        # 1 Year Return
        year_1_start = current_date - timedelta(days=365)
        year_1_data = prices[prices['Dates'] >= year_1_start]
        if len(year_1_data) > 1:
            return_1y = ((year_1_data[fund_ticker].iloc[-1] / year_1_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_1y = 0
        
        # 2024 Return
        year_2024_start = pd.to_datetime('2024-01-01')
        year_2024_end = pd.to_datetime('2024-12-31')
        year_2024_data = prices[(prices['Dates'] >= year_2024_start) & (prices['Dates'] <= year_2024_end)]
        if len(year_2024_data) > 1:
            return_2024 = ((year_2024_data[fund_ticker].iloc[-1] / year_2024_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2024 = 0
        
        # 2023 Return
        year_2023_start = pd.to_datetime('2023-01-01')
        year_2023_end = pd.to_datetime('2023-12-31')
        year_2023_data = prices[(prices['Dates'] >= year_2023_start) & (prices['Dates'] <= year_2023_end)]
        if len(year_2023_data) > 1:
            return_2023 = ((year_2023_data[fund_ticker].iloc[-1] / year_2023_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2023 = 0
        
        # 2022 Return
        year_2022_start = pd.to_datetime('2022-01-01')
        year_2022_end = pd.to_datetime('2022-12-31')
        year_2022_data = prices[(prices['Dates'] >= year_2022_start) & (prices['Dates'] <= year_2022_end)]
        if len(year_2022_data) > 1:
            return_2022 = ((year_2022_data[fund_ticker].iloc[-1] / year_2022_data[fund_ticker].iloc[0]) - 1) * 100
        else:
            return_2022 = 0
        
        # Max Drawdown
        cumulative = (1 + prices['Returns'].fillna(0)).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Volatility (annualized)
        volatility = prices['Returns'].std() * np.sqrt(252) * 100
        
        # VaR and CVaR at 5% confidence level (daily, then annualized)
        daily_returns = prices['Returns'].dropna()
        var_5_annual = 0
        cvar_5_annual = 0
        
        if len(daily_returns) >= 20:  # Need sufficient data for VaR
            # Sort returns for percentile calculation
            sorted_returns = np.sort(daily_returns)
            
            # VaR at 5% (5th percentile of losses)
            var_5_daily = np.percentile(sorted_returns, 5)
            var_5_annual = var_5_daily * np.sqrt(252) * 100  # Annualized percentage
            
            # CVaR at 5% (expected shortfall - average of worst 5% returns)
            var_index = int(0.05 * len(sorted_returns))
            if var_index > 0:
                cvar_5_daily = sorted_returns[:var_index].mean()
                cvar_5_annual = cvar_5_daily * np.sqrt(252) * 100  # Annualized percentage
        
        return {
            'YTD Return (%)': ytd_return,
            'MTD Return (%)': mtd_return,
            'Monthly Return (%)': monthly_return,
            '1Y Return (%)': return_1y,
            '2024 Return (%)': return_2024,
            '2023 Return (%)': return_2023,
            '2022 Return (%)': return_2022,
            'Max Drawdown (%)': max_drawdown,
            'Volatility (%)': volatility,
            'VaR 5% (%)': var_5_annual,
            'CVaR 5% (%)': cvar_5_annual
        }
        
    except Exception as e:
        st.error(f"Error calculating metrics for {fund_ticker}: {e}")
        return None

def calculate_custom_score(df_performance, weights):
    """Calculate custom score based on user-defined weights using Z-score standardization"""
    try:
        df_scored = df_performance.copy()
        
        # Define metrics and their direction (positive = higher is better)
        metrics_to_score = {
            'YTD Return (%)': 'positive',
            'MTD Return (%)': 'positive', 
            'Monthly Return (%)': 'positive',
            '1Y Return (%)': 'positive',
            '2024 Return (%)': 'positive',
            '2023 Return (%)': 'positive',
            '2022 Return (%)': 'positive',
            'Max Drawdown (%)': 'negative',  # Lower is better
            'Volatility (%)': 'negative',    # Lower is better
            'VaR 5% (%)': 'negative',        # Lower is better (less negative)
            'CVaR 5% (%)': 'negative'        # Lower is better (less negative)
        }
        
        # Calculate Z-scores for each metric
        z_scores = {}
        for metric, direction in metrics_to_score.items():
            if metric in df_scored.columns:
                metric_values = df_scored[metric].dropna()
                
                if len(metric_values) > 1:
                    mean_val = metric_values.mean()
                    std_val = metric_values.std()
                    
                    if std_val > 0:
                        # Calculate Z-score: (value - mean) / std
                        z_score = (df_scored[metric] - mean_val) / std_val
                        
                        # For negative metrics, invert the Z-score (lower values get higher scores)
                        if direction == 'negative':
                            z_score = -z_score
                        
                        z_scores[metric] = z_score
                    else:
                        # If no variation, assign neutral score (0)
                        z_scores[metric] = pd.Series([0] * len(df_scored), index=df_scored.index)
                else:
                    # If only one value, assign neutral score
                    z_scores[metric] = pd.Series([0] * len(df_scored), index=df_scored.index)
        
        # Calculate weighted composite score
        scores = []
        
        for idx, row in df_scored.iterrows():
            total_score = 0
            total_weight = 0
            
            for metric in metrics_to_score.keys():
                if metric in weights and weights[metric] > 0 and metric in z_scores:
                    weight = weights[metric] / 100  # Convert percentage to decimal
                    z_value = z_scores[metric].loc[idx]
                    
                    if pd.notnull(z_value):
                        total_score += z_value * weight
                        total_weight += weight
            
            # Calculate final score (weighted average of Z-scores)
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0  # Neutral score if no weights
            
            scores.append(final_score)
        
        df_scored['Custom Score'] = scores
        
        # Sort by score (higher is better)
        return df_scored.sort_values('Custom Score', ascending=False)
        
    except Exception as e:
        st.error(f"Error calculating custom score: {e}")
        return df_performance

def create_cumulative_returns_chart(funds_df, selected_funds, start_date, end_date):
    """Create interactive cumulative returns chart with custom date range"""
    try:
        # Filter data by date range
        funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
        filtered_data = funds_df[(funds_df['Dates'] >= start_date) & (funds_df['Dates'] <= end_date)].copy()
        
        if filtered_data.empty:
            return None
        
        fig = go.Figure()
        
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', 
                 '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#6366f1']
        
        for i, fund in enumerate(selected_funds):
            if fund in filtered_data.columns:
                # Calculate cumulative returns (base 100)
                prices = filtered_data[['Dates', fund]].dropna()
                if len(prices) > 1:
                    base_price = prices[fund].iloc[0]
                    cumulative_returns = (prices[fund] / base_price) * 100
                    
                    fig.add_trace(go.Scatter(
                        x=prices['Dates'],
                        y=cumulative_returns,
                        mode='lines',
                        name=fund,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Value: %{y:.2f}<br>' +
                                    '<extra></extra>'
                    ))
        
        fig.update_layout(
            title=f"Cumulative Returns Evolution (Base 100)<br><sub>From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</sub>",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (Base 100)",
            hovermode='x unified',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def create_efficient_frontier_chart(funds_df, selected_funds, fund_names_dict, performance_df, debug_container=None):
    """Create efficient frontier chart using portfolio optimization"""
    try:
        if debug_container:
            debug_container.write("🔍 **Iniciando cálculo de frontera eficiente...**")
        
        # Prepare data
        funds_df['Dates'] = pd.to_datetime(funds_df['Dates'])
        funds_subset = funds_df[['Dates'] + selected_funds].copy()
        funds_subset = funds_subset.dropna()
        
        if len(funds_subset) < 2:
            if debug_container:
                debug_container.write("❌ Datos insuficientes después de eliminar valores nulos")
            return None
        
        if debug_container:
            debug_container.write(f"✅ Datos preparados: {len(funds_subset)} observaciones")
        
        # Calculate returns
        funds_subset = funds_subset.set_index('Dates')
        returns = funds_subset.pct_change().dropna()
        
        if len(returns) < 10:
            if debug_container:
                debug_container.write("❌ Datos de retornos insuficientes")
            return None
        
        if debug_container:
            debug_container.write(f"✅ Retornos calculados: {len(returns)} observaciones")
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252  # Annualized returns
        Sigma = returns.cov() * 252  # Annualized covariance
        
        n = len(selected_funds)
        
        if debug_container:
            debug_container.write(f"✅ Parámetros calculados para {n} fondos")
            debug_container.write(f"   - Retorno promedio anualizado: {mu.mean():.2%}")
            debug_container.write(f"   - Volatilidad promedio anualizada: {np.sqrt(np.diag(Sigma)).mean():.2%}")
        
        # Generate efficient frontier
        target_returns = np.linspace(mu.min(), mu.max(), 50)
        efficient_portfolios = []
        
        for target_return in target_returns:
            try:
                # Portfolio optimization variables
                w = cp.Variable(n)
                
                # Objective: minimize portfolio variance
                portfolio_variance = cp.quad_form(w, Sigma.values)
                
                # Constraints
                constraints = [
                    cp.sum(w) == 1,  # Weights sum to 1
                    w >= 0,  # Long-only constraint
                    mu.values @ w == target_return  # Target return constraint
                ]
                
                # Solve optimization problem
                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                prob.solve(solver=cp.ECOS, verbose=False)
                
                if prob.status == cp.OPTIMAL:
                    portfolio_return = target_return
                    portfolio_risk = np.sqrt(prob.value)
                    weights = w.value
                    
                    efficient_portfolios.append({
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'weights': weights
                    })
                    
            except Exception as e:
                if debug_container:
                    debug_container.write(f"⚠️ Error en optimización para retorno {target_return:.2%}: {e}")
                continue
        
        if not efficient_portfolios:
            if debug_container:
                debug_container.write("❌ No se pudieron calcular portafolios eficientes")
            return None
        
        if debug_container:
            debug_container.write(f"✅ Frontera eficiente calculada: {len(efficient_portfolios)} puntos")
        
        # Create the plot
        fig = go.Figure()
        
        # Plot efficient frontier
        frontier_returns = [p['return'] for p in efficient_portfolios]
        frontier_risks = [p['risk'] for p in efficient_portfolios]
        
        fig.add_trace(go.Scatter(
            x=frontier_risks,
            y=frontier_returns,
            mode='lines',
            name='Frontera Eficiente',
            line=dict(color='blue', width=3),
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Plot individual funds
        individual_returns = []
        individual_risks = []
        fund_labels = []
        
        for fund in selected_funds:
            fund_return = mu[fund]
            fund_risk = np.sqrt(Sigma.loc[fund, fund])
            
            individual_returns.append(fund_return)
            individual_risks.append(fund_risk)
            
            # Get fund name from dictionary or use ticker
            fund_name = fund_names_dict.get(fund, fund)
            fund_labels.append(fund_name)
        
        fig.add_trace(go.Scatter(
            x=individual_risks,
            y=individual_returns,
            mode='markers',
            name='Fondos Individuales',
            marker=dict(size=10, color='red'),
            text=fund_labels,
            hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Find and highlight optimal portfolio (maximum Sharpe ratio)
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratios = [(p['return'] - risk_free_rate) / p['risk'] for p in efficient_portfolios]
        max_sharpe_idx = np.argmax(sharpe_ratios)
        optimal_portfolio = efficient_portfolios[max_sharpe_idx]
        
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['risk']],
            y=[optimal_portfolio['return']],
            mode='markers',
            name='Portafolio Óptimo (Max Sharpe)',
            marker=dict(size=15, color='green', symbol='star'),
            hovertemplate='Optimal Portfolio<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %.3f<extra></extra>' % sharpe_ratios[max_sharpe_idx]
        ))
        
        fig.update_layout(
            title='Frontera Eficiente de Portafolios',
            xaxis_title='Riesgo (Volatilidad Anualizada)',
            yaxis_title='Retorno Esperado Anualizado',
            hovermode='closest',
            height=600,
            showlegend=True
        )
        
        # Format axes as percentages
        fig.update_xaxes(tickformat='.1%')
        fig.update_yaxes(tickformat='.1%')
        
        if debug_container:
            debug_container.write("✅ Gráfico de frontera eficiente creado exitosamente")
            debug_container.write(f"   - Portafolio óptimo: Retorno {optimal_portfolio['return']:.2%}, Riesgo {optimal_portfolio['risk']:.2%}")
            debug_container.write(f"   - Ratio de Sharpe óptimo: {sharpe_ratios[max_sharpe_idx]:.3f}")
        
        return fig, optimal_portfolio
        
    except Exception as e:
        if debug_container:
            debug_container.write(f"❌ Error general en cálculo de frontera eficiente: {e}")
        st.error(f"Error creating efficient frontier: {e}")
        return None

def generate_pdf_report(performance_df, selected_funds, fund_names_dict):
    """Generate PDF report with fund analysis"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph("Fund Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Date
        date_style = ParagraphStyle(
            'DateStyle',
            parent=styles['Normal'],
            fontSize=12,
            alignment=1
        )
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", date_style))
        story.append(Spacer(1, 30))
        
        # Summary
        summary_style = ParagraphStyle(
            'SummaryStyle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12
        )
        story.append(Paragraph("Executive Summary", summary_style))
        
        summary_text = f"""
        This report analyzes {len(selected_funds)} selected funds based on various performance metrics.
        The analysis includes returns across different time periods, risk metrics, and custom scoring.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Performance table
        story.append(Paragraph("Performance Metrics", summary_style))
        
        # Prepare table data
        table_data = [['Fund', 'YTD Return', '1Y Return', 'Volatility', 'Max Drawdown', 'Custom Score']]
        
        for _, row in performance_df.head(10).iterrows():  # Top 10 funds
            fund_name = fund_names_dict.get(row['Ticker'], row['Ticker'])
            table_data.append([
                fund_name[:30] + "..." if len(fund_name) > 30 else fund_name,
                f"{row['YTD Return (%)']:.2f}%",
                f"{row['1Y Return (%)']:.2f}%",
                f"{row['Volatility (%)']:.2f}%",
                f"{row['Max Drawdown (%)']:.2f}%",
                f"{row.get('Custom Score', 0):.3f}"
            ])
        
        # Create table
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Methodology
        story.append(Paragraph("Methodology", summary_style))
        methodology_text = """
        The analysis uses the following metrics:
        • YTD Return: Year-to-date performance
        • 1Y Return: One-year rolling return
        • Volatility: Annualized standard deviation of returns
        • Max Drawdown: Maximum peak-to-trough decline
        • VaR/CVaR: Value at Risk and Conditional Value at Risk at 5% confidence level
        • Custom Score: Z-score weighted composite ranking
        """
        story.append(Paragraph(methodology_text, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Dashboard de Análisis de Fondos</h1>', unsafe_allow_html=True)
    
    # Load data
    funds_data, etf_dict = load_data()
    
    if funds_data is None or etf_dict is None:
        st.stop()
    
    # Get available funds
    available_funds = [col for col in funds_data.columns if col != 'Dates']
    
    # SIDEBAR - Filters
    st.sidebar.markdown("# 🔍 Filters")
    
    # Create mapping from ETF dictionary
    fund_info = {}
    if not etf_dict.empty:
        for _, row in etf_dict.iterrows():
            dict_ticker = str(row.get('Ticker', ''))
            
            # Try to match with available funds
            matched_fund = None
            for fund in available_funds:
                if fund == dict_ticker:
                    matched_fund = fund
                    break
                if fund.replace(' ', '').upper() == dict_ticker.replace(' ', '').upper():
                    matched_fund = fund
                    break
                if dict_ticker.replace(' ', '').upper() in fund.replace(' ', '').upper():
                    matched_fund = fund
                    break
            
            if matched_fund:
                fund_info[matched_fund] = {
                    'name': row.get('Indice', matched_fund),
                    'region': row.get('Geografia', 'Unknown'),
                    'asset_class': row.get('Asset Class', 'Unknown'),
                    'subclass': row.get('Subclass', 'Unknown'),
                    'sector': row.get('Sector', 'Unknown'),
                    'original_ticker': dict_ticker
                }
    
    # Get unique categories for filters (all available initially)
    all_regions = list(set([info.get('region', 'Unknown') for info in fund_info.values()]))
    all_asset_classes = list(set([info.get('asset_class', 'Unknown') for info in fund_info.values()]))
    all_subclasses = list(set([info.get('subclass', 'Unknown') for info in fund_info.values()]))
    all_sectors = list(set([info.get('sector', 'Unknown') for info in fund_info.values()]))
    
    # Remove 'Unknown' if there are other options
    if len(all_regions) > 1 and 'Unknown' in all_regions:
        all_regions.remove('Unknown')
    if len(all_asset_classes) > 1 and 'Unknown' in all_asset_classes:
        all_asset_classes.remove('Unknown')
    if len(all_subclasses) > 1 and 'Unknown' in all_subclasses:
        all_subclasses.remove('Unknown')
    if len(all_sectors) > 1 and 'Unknown' in all_sectors:
        all_sectors.remove('Unknown')
    
    # Data summary
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.write(f"📈 Total funds: {len(available_funds)}")
    st.sidebar.write(f"🏷️ With metadata: {len(fund_info)}")
    st.sidebar.write(f"🌍 Regions: {len(all_regions)}")
    st.sidebar.write(f"💼 Asset classes: {len(all_asset_classes)}")
    st.sidebar.write(f"🏷️ Subclasses: {len(all_subclasses)}")
    st.sidebar.write(f"🏭 Sectors: {len(all_sectors)}")
    
    # Filter controls with cascading logic
    st.sidebar.markdown("### 🎯 Filter Controls")
    
    # Region filter (always shows all regions)
    selected_regions = st.sidebar.multiselect(
        "Seleccionar Regiones:",
        options=sorted(all_regions),
        default=[]  # Start with no selection
    )
    
    # Filter available asset classes based on selected regions
    if selected_regions:
        available_asset_classes = list(set([
            info.get('asset_class', 'Unknown') 
            for info in fund_info.values() 
            if info.get('region', 'Unknown') in selected_regions
        ]))
        if 'Unknown' in available_asset_classes and len(available_asset_classes) > 1:
            available_asset_classes.remove('Unknown')
    else:
        available_asset_classes = all_asset_classes
    
    # Asset class filter
    selected_asset_classes = st.sidebar.multiselect(
        "Seleccionar Clases de Activos:",
        options=sorted(available_asset_classes),
        default=[]  # Start with no selection
    )
    
    # Filter available subclasses based on selected regions and asset classes
    if selected_regions or selected_asset_classes:
        available_subclasses = list(set([
            info.get('subclass', 'Unknown') 
            for info in fund_info.values() 
            if (not selected_regions or info.get('region', 'Unknown') in selected_regions) and
               (not selected_asset_classes or info.get('asset_class', 'Unknown') in selected_asset_classes)
        ]))
        if 'Unknown' in available_subclasses and len(available_subclasses) > 1:
            available_subclasses.remove('Unknown')
    else:
        available_subclasses = all_subclasses
    
    # Subclass filter
    selected_subclasses = st.sidebar.multiselect(
        "Seleccionar Subclases:",
        options=sorted(available_subclasses),
        default=[]  # Start with no selection
    )
    
    # Filter available sectors based on all previous selections
    if selected_regions or selected_asset_classes or selected_subclasses:
        available_sectors = list(set([
            info.get('sector', 'Unknown') 
            for info in fund_info.values() 
            if (not selected_regions or info.get('region', 'Unknown') in selected_regions) and
               (not selected_asset_classes or info.get('asset_class', 'Unknown') in selected_asset_classes) and
               (not selected_subclasses or info.get('subclass', 'Unknown') in selected_subclasses)
        ]))
        if 'Unknown' in available_sectors and len(available_sectors) > 1:
            available_sectors.remove('Unknown')
    else:
        available_sectors = all_sectors
    
    # Sector filter
    selected_sectors = st.sidebar.multiselect(
        "Seleccionar Sectores:",
        options=sorted(available_sectors),
        default=[]  # Start with no selection
    )
    
    # Apply filters to get filtered funds
    filtered_funds = []
    
    if not any([selected_regions, selected_asset_classes, selected_subclasses, selected_sectors]):
        # If no filters selected, show all funds
        filtered_funds = available_funds
    else:
        # Apply filters
        for fund in available_funds:
            if fund in fund_info:
                info = fund_info[fund]
                
                # Check if fund matches all selected filters
                region_match = not selected_regions or info.get('region', 'Unknown') in selected_regions
                asset_match = not selected_asset_classes or info.get('asset_class', 'Unknown') in selected_asset_classes
                subclass_match = not selected_subclasses or info.get('subclass', 'Unknown') in selected_subclasses
                sector_match = not selected_sectors or info.get('sector', 'Unknown') in selected_sectors
                
                if region_match and asset_match and subclass_match and sector_match:
                    filtered_funds.append(fund)
            else:
                # Include funds without metadata if no specific filters are applied
                if not any([selected_regions, selected_asset_classes, selected_subclasses, selected_sectors]):
                    filtered_funds.append(fund)
    
    st.sidebar.markdown("### 📊 Filtered Results")
    st.sidebar.write(f"🎯 Filtered funds: {len(filtered_funds)}")
    
    # MAIN CONTENT
    if filtered_funds:
        # Calculate performance metrics for filtered funds
        st.markdown("## 📊 Performance Analysis")
        
        with st.spinner("Calculating performance metrics..."):
            performance_data = []
            
            progress_bar = st.progress(0)
            for i, fund in enumerate(filtered_funds):
                metrics = calculate_performance_metrics(funds_data, fund)
                if metrics:
                    fund_name = fund_info.get(fund, {}).get('name', fund)
                    
                    performance_data.append({
                        'Ticker': fund,
                        'Fund Name': fund_name,
                        **metrics
                    })
                
                progress_bar.progress((i + 1) / len(filtered_funds))
            
            progress_bar.empty()
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            
            # Custom scoring section
            st.markdown("### 🎯 Custom Scoring")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("#### Weight Configuration")
                st.markdown("*Adjust weights for custom ranking (total should be 100%)*")
                
                weights = {}
                metrics_for_scoring = [
                    'YTD Return (%)', 'MTD Return (%)', 'Monthly Return (%)', '1Y Return (%)',
                    '2024 Return (%)', '2023 Return (%)', '2022 Return (%)',
                    'Max Drawdown (%)', 'Volatility (%)', 'VaR 5% (%)', 'CVaR 5% (%)'
                ]
                
                # Default weights
                default_weights = {
                    'YTD Return (%)': 20,
                    '1Y Return (%)': 25,
                    '2024 Return (%)': 15,
                    'Max Drawdown (%)': 15,
                    'Volatility (%)': 15,
                    'VaR 5% (%)': 10
                }
                
                total_weight = 0
                for metric in metrics_for_scoring:
                    default_val = default_weights.get(metric, 0)
                    weight = st.slider(
                        metric.replace(' (%)', ''),
                        min_value=0,
                        max_value=50,
                        value=default_val,
                        step=1,
                        key=f"weight_{metric}"
                    )
                    weights[metric] = weight
                    total_weight += weight
                
                st.write(f"**Total Weight: {total_weight}%**")
                if total_weight != 100:
                    st.warning("⚠️ Weights should sum to 100% for optimal scoring")
            
            with col1:
                # Calculate custom scores
                df_scored = calculate_custom_score(df_performance, weights)
                
                st.markdown("#### Top Performing Funds")
                
                # Display top funds
                display_columns = ['Ticker', 'Fund Name', 'YTD Return (%)', '1Y Return (%)', 
                                 'Max Drawdown (%)', 'Volatility (%)', 'Custom Score']
                
                # Format the display dataframe
                df_display = df_scored[display_columns].copy()
                
                # Format numeric columns
                numeric_columns = ['YTD Return (%)', '1Y Return (%)', 'Max Drawdown (%)', 'Volatility (%)']
                for col in numeric_columns:
                    df_display[col] = df_display[col].round(2)
                
                df_display['Custom Score'] = df_display['Custom Score'].round(3)
                
                st.dataframe(
                    df_display.head(20),
                    use_container_width=True,
                    hide_index=True
                )
            
            # Charts section
            st.markdown("### 📈 Performance Charts")
            
            # Fund selection for charts
            top_funds = df_scored.head(10)['Ticker'].tolist()
            
            selected_funds_for_chart = st.multiselect(
                "Select funds for comparison (max 10):",
                options=filtered_funds,
                default=top_funds[:5],
                max_selections=10
            )
            
            if selected_funds_for_chart:
                # Date range selection
                col1, col2 = st.columns(2)
                
                funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
                min_date = funds_data['Dates'].min().date()
                max_date = funds_data['Dates'].max().date()
                
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=max_date - timedelta(days=365),
                        min_value=min_date,
                        max_value=max_date
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date
                    )
                
                # Create cumulative returns chart
                if start_date < end_date:
                    chart = create_cumulative_returns_chart(
                        funds_data, 
                        selected_funds_for_chart, 
                        pd.to_datetime(start_date), 
                        pd.to_datetime(end_date)
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning("No data available for the selected date range and funds.")
                else:
                    st.error("Start date must be before end date.")
            
            # Efficient Frontier Analysis
            st.markdown("### 🎯 Efficient Frontier Analysis")
            
            selected_funds_for_frontier = st.multiselect(
                "Select funds for efficient frontier analysis (2-10 funds):",
                options=filtered_funds,
                default=top_funds[:5] if len(top_funds) >= 5 else top_funds,
                max_selections=10
            )
            
            if len(selected_funds_for_frontier) >= 2:
                with st.spinner("Calculating efficient frontier..."):
                    # Create fund names dictionary
                    fund_names_dict = {}
                    for fund in selected_funds_for_frontier:
                        fund_names_dict[fund] = fund_info.get(fund, {}).get('name', fund)
                    
                    # Get performance data for selected funds
                    selected_performance = df_scored[df_scored['Ticker'].isin(selected_funds_for_frontier)]
                    
                    # Create debug container
                    debug_container = st.expander("🔍 Debug Information", expanded=False)
                    
                    result = create_efficient_frontier_chart(
                        funds_data, 
                        selected_funds_for_frontier, 
                        fund_names_dict, 
                        selected_performance,
                        debug_container
                    )
                    
                    if result and len(result) == 2:
                        fig, optimal_portfolio = result
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show optimal portfolio composition
                        st.markdown("#### Optimal Portfolio Composition")
                        
                        if optimal_portfolio and 'weights' in optimal_portfolio:
                            weights_df = pd.DataFrame({
                                'Fund': selected_funds_for_frontier,
                                'Weight (%)': [w * 100 for w in optimal_portfolio['weights']],
                                'Fund Name': [fund_names_dict.get(fund, fund) for fund in selected_funds_for_frontier]
                            })
                            
                            # Filter out very small weights
                            weights_df = weights_df[weights_df['Weight (%)'] > 0.1].sort_values('Weight (%)', ascending=False)
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.dataframe(
                                    weights_df[['Fund Name', 'Weight (%)']].round(2),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            with col2:
                                # Pie chart of portfolio composition
                                fig_pie = px.pie(
                                    weights_df,
                                    values='Weight (%)',
                                    names='Fund Name',
                                    title='Portfolio Composition'
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                    
                    elif result:
                        st.plotly_chart(result, use_container_width=True)
            else:
                st.info("Select at least 2 funds to see efficient frontier analysis.")
            
            # Export functionality
            st.markdown("### 📄 Export Report")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                csv_data = df_scored.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV Report",
                    data=csv_data,
                    file_name=f"fund_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # PDF export
                if st.button("📄 Generate PDF Report"):
                    with st.spinner("Generating PDF report..."):
                        fund_names_dict = {}
                        for fund in filtered_funds:
                            fund_names_dict[fund] = fund_info.get(fund, {}).get('name', fund)
                        
                        pdf_buffer = generate_pdf_report(df_scored, selected_funds_for_chart, fund_names_dict)
                        
                        if pdf_buffer:
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"fund_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Failed to generate PDF report")
        
        else:
            st.warning("No performance data available for the filtered funds.")
    
    else:
        st.warning("No funds match the selected filters. Please adjust your filter criteria.")

if __name__ == "__main__":
    main()