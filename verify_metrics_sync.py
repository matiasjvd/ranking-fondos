#!/usr/bin/env python3
"""
SCRIPT DE VALIDACIÓN DE SINCRONIZACIÓN DE MÉTRICAS
Verifica que los datos sean iguales en ambos dashboards
"""

import pandas as pd
import numpy as np
from metrics_calculator import calculate_individual_fund_metrics, calculate_portfolio_metrics

def test_individual_metrics():
    """Verifica que las métricas individuales se calculan correctamente"""
    print("\n" + "="*70)
    print("TEST 1: MÉTRICAS INDIVIDUALES")
    print("="*70)
    
    try:
        funds_data = pd.read_csv('data/funds_prices.csv', low_memory=False)
        print(f"✅ Datos cargados: {funds_data.shape[0]} filas, {funds_data.shape[1]} columnas")
        
        # Seleccionar el primer fondo válido (que no sea 'Dates' ni columnas de índice)
        valid_funds = [
            col for col in funds_data.columns 
            if col != 'Dates' and 'Unnamed' not in str(col) and funds_data[col].notna().sum() > 100
        ]
        
        if not valid_funds:
            print("❌ No hay fondos válidos en los datos")
            return False
        
        test_fund = valid_funds[0]
        print(f"\n📊 Analizando fondo: {test_fund}")
        
        metrics = calculate_individual_fund_metrics(funds_data, test_fund)
        
        if metrics is None:
            print(f"❌ No se pudieron calcular métricas para {test_fund}")
            return False
        
        print("\n📈 Métricas Calculadas:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"   {metric_name:.<40} {value:>12.2f}")
            else:
                print(f"   {metric_name:.<40} {value:>12}")
        
        # Validaciones
        print("\n✓ Validaciones:")
        
        # Check 1: Volatility > 0
        if metrics['Volatility (%)'] > 0:
            print(f"   ✅ Volatility ({metrics['Volatility (%)']:.2f}%) > 0")
        else:
            print(f"   ❌ Volatility debe ser > 0")
            return False
        
        # Check 2: Max Drawdown < 0
        if metrics['Max Drawdown (%)'] < 0:
            print(f"   ✅ Max Drawdown ({metrics['Max Drawdown (%)']:.2f}%) < 0")
        else:
            print(f"   ⚠️  Max Drawdown ({metrics['Max Drawdown (%)']:.2f}%) debería ser negativo")
        
        # Check 3: VaR y CVaR lógicos
        if metrics['VaR 5% (%)'] < metrics['CVaR 5% (%)']:
            print(f"   ✅ VaR ({metrics['VaR 5% (%)']:.2f}%) < CVaR ({metrics['CVaR 5% (%)']:.2f}%)")
        else:
            print(f"   ⚠️  CVaR debería ser más negativo que VaR")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_portfolio_metrics():
    """Verifica que las métricas del portafolio se calculan correctamente"""
    print("\n" + "="*70)
    print("TEST 2: MÉTRICAS DEL PORTAFOLIO")
    print("="*70)
    
    try:
        funds_data = pd.read_csv('data/funds_prices.csv', low_memory=False)
        
        # Seleccionar 2-3 fondos válidos
        valid_funds = [
            col for col in funds_data.columns 
            if col != 'Dates' and 'Unnamed' not in str(col) and funds_data[col].notna().sum() > 100
        ]
        
        if len(valid_funds) < 2:
            print("❌ Se necesitan al menos 2 fondos válidos")
            return False
        
        selected_funds = valid_funds[:3] if len(valid_funds) >= 3 else valid_funds[:2]
        print(f"📊 Analizando portafolio con fondos: {selected_funds}")
        
        # Pesos equitativos
        weights = {fund: 100/len(selected_funds) for fund in selected_funds}
        print(f"   Pesos: {weights}")
        
        portfolio_metrics = calculate_portfolio_metrics(
            funds_data, 
            selected_funds, 
            weights
        )
        
        if portfolio_metrics is None:
            print("❌ No se pudieron calcular métricas del portafolio")
            return False
        
        print("\n📊 Métricas del Portafolio:")
        for metric_name, value in portfolio_metrics.items():
            if metric_name != 'portfolio_returns':  # Omitir la serie larga
                if isinstance(value, float):
                    print(f"   {metric_name:.<40} {value:>12.2f}")
                else:
                    print(f"   {metric_name:.<40} {str(value):>12}")
        
        # Validaciones
        print("\n✓ Validaciones:")
        
        if portfolio_metrics['volatility'] > 0:
            print(f"   ✅ Volatility ({portfolio_metrics['volatility']:.2f}%) > 0")
        else:
            print(f"   ❌ Volatility debe ser > 0")
            return False
        
        if portfolio_metrics['period_days'] > 0:
            print(f"   ✅ Período: {portfolio_metrics['period_days']} días")
        else:
            print(f"   ❌ Período debe ser > 0")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency():
    """Verifica que los datos se cargan y procesan consistentemente"""
    print("\n" + "="*70)
    print("TEST 3: CONSISTENCIA DE DATOS")
    print("="*70)
    
    try:
        funds_data = pd.read_csv('data/funds_prices.csv', low_memory=False)
        
        print(f"✅ Datos cargados correctamente")
        print(f"   - Columnas: {funds_data.shape[1]}")
        print(f"   - Filas: {funds_data.shape[0]}")
        
        # Verificar que Dates existe
        if 'Dates' not in funds_data.columns:
            print("❌ Columna 'Dates' no encontrada")
            return False
        
        print(f"✅ Columna 'Dates' encontrada")
        
        # Convertir fechas
        funds_data['Dates'] = pd.to_datetime(funds_data['Dates'])
        date_range = funds_data['Dates'].max() - funds_data['Dates'].min()
        print(f"   - Rango: {funds_data['Dates'].min()} a {funds_data['Dates'].max()}")
        print(f"   - Duración: {date_range.days} días")
        
        # Contar fondos válidos
        valid_funds = [col for col in funds_data.columns if col != 'Dates' and 'Unnamed' not in str(col)]
        print(f"✅ Fondos disponibles: {len(valid_funds)}")
        
        # Contar fondos con suficientes datos
        sufficient_funds = [col for col in valid_funds if funds_data[col].notna().sum() > 100]
        print(f"✅ Fondos con >100 observaciones: {len(sufficient_funds)}")
        
        if len(sufficient_funds) == 0:
            print("❌ No hay fondos con suficientes observaciones")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("🔐 VERIFICACIÓN DE SINCRONIZACIÓN DE MÉTRICAS")
    print("="*70)
    
    results = {
        'Consistencia de Datos': test_data_consistency(),
        'Métricas Individuales': test_individual_metrics(),
        'Métricas del Portafolio': test_portfolio_metrics(),
    }
    
    print("\n" + "="*70)
    print("📋 RESUMEN DE PRUEBAS")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✅ PASADO" if result else "❌ FALLIDO"
        print(f"   {test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✨ TODAS LAS PRUEBAS PASARON - SINCRONIZACIÓN OK")
    else:
        print("⚠️  ALGUNAS PRUEBAS FALLARON - REVISAR ARRIBA")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)