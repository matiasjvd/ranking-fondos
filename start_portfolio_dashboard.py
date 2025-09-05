#!/usr/bin/env python3
"""
PORTFOLIO DASHBOARD LAUNCHER
Script para ejecutar el dashboard con funcionalidad de portafolios
"""

import subprocess
import sys
import os

def main():
    """Ejecutar el dashboard con portafolios"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(script_dir, 'dashboard_with_portfolio.py')
        
        print("🚀 Iniciando Dashboard de Fondos con Constructor de Portafolios...")
        print("\n📊 Funcionalidades disponibles:")
        print("   ✅ Análisis completo de fondos")
        print("   ✅ Constructor de portafolios interactivo")
        print("   ✅ Carrito de fondos en tiempo real")
        print("   ✅ Métricas de riesgo y retorno")
        print("   ✅ Simulaciones históricas")
        print("   ✅ Exportación a Excel")
        print("\n🌐 Abriendo en el navegador...")
        print("📍 URL: http://localhost:8502")
        
        # Ejecutar Streamlit en puerto específico
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8502",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()