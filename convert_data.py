#!/usr/bin/env python3
"""
Script para convertir/actualizar los datos a CSV para el proyecto de ranking de fondos.
- Actualiza el diccionario desde un Excel local
- Fusiona/añade nuevas series de precios desde un Excel adicional, alineando por 'Dates'

Requisitos: pandas, openpyxl
"""

import os
import pandas as pd
from typing import Dict, Optional


# === Rutas por defecto (puedes ajustarlas si cambian) ===
# Excel con nuevas acciones y precios (provisto por el usuario)
DEFAULT_NEW_PRICES_XLSX = \
    '/Users/matias/Desktop/Proyectos/ranking-fondos/Swiss_stocks_10-09-2025.xlsx'

# Excel del diccionario actualizado (provisto por el usuario)
DEFAULT_DICT_XLSX = \
    '/Users/matias/Desktop/Proyectos/ranking-fondos/dict_temp_full_portfolio.xlsx'

# CSVs de salida usados por el dashboard
OUTPUT_DIR = '/Users/matias/Desktop/Proyectos/ranking-fondos/data'
PRICES_CSV = os.path.join(OUTPUT_DIR, 'funds_prices.csv')
DICT_CSV = os.path.join(OUTPUT_DIR, 'funds_dictionary.csv')


def _normalize_dates(df: pd.DataFrame, date_col: str = 'Dates') -> pd.DataFrame:
    """Normaliza columna 'Dates' a datetime y ordena."""
    if date_col not in df.columns:
        return df

    s = df[date_col]
    # Intento 1: parseo directo
    dates = pd.to_datetime(s, errors='coerce')

    # Si casi todo es NaT, puede que sean seriales de Excel (números)
    nat_ratio = dates.isna().mean()
    if nat_ratio > 0.8 and pd.api.types.is_numeric_dtype(s):
        # Excel date serial -> origin 1899-12-30
        dates = pd.to_datetime(s, unit='D', origin='1899-12-30', errors='coerce')

    df = df.copy()
    df[date_col] = dates
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return df


def _read_all_sheets_prices(xlsx_path: str) -> Optional[pd.DataFrame]:
    """
    Lee todas las hojas de un Excel de precios, busca 'Dates' y fusiona por esa columna.
    Devuelve un DataFrame con 'Dates' + columnas de precios o None si falla/no hay datos válidos.
    """
    try:
        xls: Dict[str, pd.DataFrame] = pd.read_excel(xlsx_path, sheet_name=None)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo de precios: {xlsx_path}")
        return None
    except Exception as e:
        print(f"❌ Error leyendo Excel de precios {xlsx_path}: {e}")
        return None

    merged: Optional[pd.DataFrame] = None
    sheets_used = []

    for sheet_name, df in xls.items():
        if df is None or df.empty:
            continue
        # Elimina columnas completamente vacías
        df = df.dropna(axis=1, how='all')
        # Detecta columna de fecha
        if 'Dates' not in df.columns:
            # Heurística: si la primera columna parece fecha, renómbrala a 'Dates'
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'Dates'})
        if 'Dates' not in df.columns:
            continue
        df = _normalize_dates(df, 'Dates')
        if df.empty:
            continue

        if merged is None:
            merged = df
        else:
            # Merge outer para no perder ninguna fecha
            merged = pd.merge(merged, df, on='Dates', how='outer')
        sheets_used.append(sheet_name)

    if merged is None or merged.empty:
        print(f"⚠️ No se encontraron hojas válidas con columna 'Dates' en: {xlsx_path}")
        return None

    # Elimina columnas completamente vacías tras el merge
    merged = merged.dropna(axis=1, how='all')
    merged = _normalize_dates(merged, 'Dates')

    print(f"✅ Leídas {len(sheets_used)} hojas desde {os.path.basename(xlsx_path)}: {', '.join(sheets_used[:5])}{'...' if len(sheets_used) > 5 else ''}")
    print(f"   - Columnas (incl. Dates): {len(merged.columns)}  |  Filas: {len(merged)}")
    return merged


def update_dictionary(dict_xlsx_path: str = DEFAULT_DICT_XLSX) -> bool:
    """Genera/actualiza el CSV del diccionario desde el Excel provisto."""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        etf_dict = pd.read_excel(dict_xlsx_path)

        # Limpieza básica opcional del ticker (recorta espacios al borde)
        if 'Ticker' in etf_dict.columns:
            etf_dict['Ticker'] = etf_dict['Ticker'].astype(str).str.strip()

        etf_dict.to_csv(DICT_CSV, index=False)
        print(f"✅ Diccionario actualizado: {DICT_CSV}")
        print(f"   - Columnas: {list(etf_dict.columns)}")
        print(f"   - Filas: {len(etf_dict)}")
        return True
    except FileNotFoundError as e:
        print(f"❌ No se encontró el Excel de diccionario: {dict_xlsx_path}")
        return False
    except Exception as e:
        print(f"❌ Error actualizando diccionario: {e}")
        return False


def merge_new_prices(new_prices_xlsx: str = DEFAULT_NEW_PRICES_XLSX,
                     existing_prices_csv: str = PRICES_CSV) -> bool:
    """
    Fusiona los precios nuevos (Excel) con el CSV existente de precios por 'Dates'.
    - Si no existe el CSV, crea uno con los datos nuevos.
    - Si existe, añade columnas nuevas y mantiene las existentes.
    """
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Lee nuevos precios desde Excel
        new_prices = _read_all_sheets_prices(new_prices_xlsx)
        if new_prices is None:
            return False

        # Lee precios existentes (si hay)
        if os.path.exists(existing_prices_csv):
            existing = pd.read_csv(existing_prices_csv)
            existing = _normalize_dates(existing, 'Dates')
            print(f"ℹ️ Precios existentes cargados: {existing_prices_csv}")
            print(f"   - Columnas: {len(existing.columns)}  |  Filas: {len(existing)}")
        else:
            existing = None

        # Merge
        if existing is None or existing.empty:
            merged = new_prices
        else:
            merged = pd.merge(existing, new_prices, on='Dates', how='outer', suffixes=('', '_NEW'))
            # Si hay columnas duplicadas con sufijo _NEW (mismo ticker), decide cómo resolver
            dup_new_cols = [c for c in merged.columns if c.endswith('_NEW')]
            for c_new in dup_new_cols:
                c = c_new[:-4]
                # Si columna original existe, rellenar NaN con valores nuevos
                if c in merged.columns:
                    merged[c] = merged[c].fillna(merged[c_new])
                    merged.drop(columns=[c_new], inplace=True)
                else:
                    # Renombrar si no existía
                    merged.rename(columns={c_new: c}, inplace=True)

        merged = _normalize_dates(merged, 'Dates')
        merged.to_csv(existing_prices_csv, index=False)

        print(f"✅ Precios actualizados: {existing_prices_csv}")
        print(f"   - Columnas: {len(merged.columns)}  |  Filas: {len(merged)}")
        if 'Dates' in merged.columns:
            try:
                print(f"   - Rango de fechas: {merged['Dates'].min()} a {merged['Dates'].max()}")
            except Exception:
                pass
        return True

    except FileNotFoundError as e:
        print(f"❌ No se encontró archivo: {e}")
        return False
    except Exception as e:
        print(f"❌ Error fusionando precios: {e}")
        return False


def convert_excel_to_csv(
    dict_xlsx_path: str = DEFAULT_DICT_XLSX,
    new_prices_xlsx: str = DEFAULT_NEW_PRICES_XLSX,
) -> bool:
    """
    Flujo principal:
    1) Actualiza diccionario desde Excel local
    2) Fusiona/añade nuevas series de precios desde Excel local con el CSV existente
    """
    ok_dict = update_dictionary(dict_xlsx_path)
    ok_prices = merge_new_prices(new_prices_xlsx)
    return ok_dict and ok_prices


if __name__ == "__main__":
    print("🔄 Iniciando actualización de datos...")
    success = convert_excel_to_csv()
    if success:
        print("\n✅ Actualización completada. Los CSV están listos para el dashboard.")
    else:
        print("\n❌ Hubo errores durante la actualización. Revisa los mensajes arriba.")