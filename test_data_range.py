#!/usr/bin/env python3
"""
Script para verificar el rango de datos en el dashboard
"""

import pandas as pd
import os

def test_data_range():
    """Verifica el rango de datos disponible"""
    
    # Cargar datos como lo hace el dashboard
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    funds_path = os.path.join(data_dir, 'funds_prices.csv')
    
    print("🔍 Verificando datos del dashboard...")
    print(f"📁 Archivo: {funds_path}")
    
    try:
        # Cargar datos
        funds = pd.read_csv(funds_path, low_memory=False)
        funds['Dates'] = pd.to_datetime(funds['Dates'])
        
        print(f"✅ Datos cargados exitosamente")
        print(f"📊 Total de filas: {len(funds):,}")
        print(f"📅 Fecha mínima: {funds['Dates'].min().strftime('%Y-%m-%d')}")
        print(f"📅 Fecha máxima: {funds['Dates'].max().strftime('%Y-%m-%d')}")
        print(f"🏛️ Total de fondos: {len([col for col in funds.columns if col != 'Dates']):,}")
        
        # Verificar datos recientes (últimos 30 días)
        recent_cutoff = funds['Dates'].max() - pd.Timedelta(days=30)
        recent_data = funds[funds['Dates'] >= recent_cutoff]
        
        print(f"\n📈 Datos recientes (últimos 30 días):")
        print(f"   Desde: {recent_cutoff.strftime('%Y-%m-%d')}")
        print(f"   Filas: {len(recent_data)}")
        
        print(f"\n📋 Últimas 10 fechas disponibles:")
        for date in funds['Dates'].tail(10):
            print(f"   {date.strftime('%Y-%m-%d')}")
            
        # Verificar si hay datos después del 10 de septiembre
        sep_10 = pd.to_datetime('2025-09-10')
        after_sep_10 = funds[funds['Dates'] > sep_10]
        
        print(f"\n🎯 Datos después del 10 de septiembre de 2025:")
        print(f"   Filas: {len(after_sep_10)}")
        if len(after_sep_10) > 0:
            print(f"   Desde: {after_sep_10['Dates'].min().strftime('%Y-%m-%d')}")
            print(f"   Hasta: {after_sep_10['Dates'].max().strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_data_range()