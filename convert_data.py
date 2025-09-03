#!/usr/bin/env python3
"""
Script para convertir los datos de Excel a CSV para el proyecto de ranking de fondos
"""

import pandas as pd
import os

def convert_excel_to_csv():
    """Convierte los archivos Excel a CSV"""
    
    # Rutas de los archivos originales
    excel_prices_path = '/Users/matias/Downloads/Prices_1970-2025.xlsx'
    excel_dict_path = '/Users/matias/Desktop/Proyectos/quant-allocation/dict_temp_full_portfolio.xlsx'
    
    # Rutas de destino
    output_dir = '/Users/matias/Desktop/Proyectos/ranking-fondos/data'
    
    # Crear directorio de datos si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("Convirtiendo datos de precios...")
        
        # Cargar y combinar las hojas de precios
        funds_1 = pd.read_excel(excel_prices_path, sheet_name='1ros')
        funds_1 = funds_1.dropna(axis=1, how='all')
        
        funds_2 = pd.read_excel(excel_prices_path, sheet_name='2dos')
        
        # Combinar los datos
        funds = pd.merge(funds_1, funds_2, on='Dates', how='outer')
        funds = funds.sort_values('Dates').reset_index(drop=True)
        
        # Guardar como CSV
        funds_csv_path = os.path.join(output_dir, 'funds_prices.csv')
        funds.to_csv(funds_csv_path, index=False)
        print(f"✅ Datos de precios guardados en: {funds_csv_path}")
        print(f"   - Columnas: {len(funds.columns)}")
        print(f"   - Filas: {len(funds)}")
        print(f"   - Rango de fechas: {funds['Dates'].min()} a {funds['Dates'].max()}")
        
        # Cargar y guardar el diccionario de ETFs
        print("\nConvirtiendo diccionario de fondos...")
        etf_dict = pd.read_excel(excel_dict_path)
        
        dict_csv_path = os.path.join(output_dir, 'funds_dictionary.csv')
        etf_dict.to_csv(dict_csv_path, index=False)
        print(f"✅ Diccionario de fondos guardado en: {dict_csv_path}")
        print(f"   - Columnas: {list(etf_dict.columns)}")
        print(f"   - Filas: {len(etf_dict)}")
        
        # Mostrar información sobre los fondos
        print(f"\n📊 Resumen de datos:")
        print(f"   - Total de fondos en precios: {len(funds.columns) - 1}")  # -1 por la columna Dates
        print(f"   - Total de fondos en diccionario: {len(etf_dict)}")
        
        # Verificar coincidencias entre precios y diccionario
        price_funds = set(funds.columns) - {'Dates'}
        dict_tickers = set(etf_dict['Ticker'].astype(str))
        
        matches = price_funds.intersection(dict_tickers)
        print(f"   - Fondos con metadata completa: {len(matches)}")
        
        if len(matches) < len(price_funds):
            missing_metadata = price_funds - dict_tickers
            print(f"   - Fondos sin metadata: {len(missing_metadata)}")
            if len(missing_metadata) <= 10:
                print(f"     {list(missing_metadata)}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
        print("Verifica que existan los siguientes archivos:")
        print(f"  - {excel_prices_path}")
        print(f"  - {excel_dict_path}")
        return False
        
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        return False

if __name__ == "__main__":
    print("🔄 Iniciando conversión de datos Excel a CSV...")
    success = convert_excel_to_csv()
    
    if success:
        print("\n✅ Conversión completada exitosamente!")
        print("Los archivos CSV están listos para usar en el dashboard.")
    else:
        print("\n❌ La conversión falló. Revisa los errores anteriores.")