#!/usr/bin/env python3
"""
LAUNCHER FOR PORTFOLIO-ENABLED DASHBOARD
Script para ejecutar el dashboard con funcionalidad de portafolios
"""

import subprocess
import sys
import os

def main():
    """Ejecutar el dashboard principal con portafolios"""
    try:
        # Obtener el directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(script_dir, 'main_dashboard.py')
        
        print("🚀 Iniciando Dashboard de Fondos + Constructor de Portafolios...")
        print("📊 Funcionalidades disponibles:")
        print("   • Análisis completo de fondos (funcionalidad original)")
        print("   • Constructor de portafolios interactivo")
        print("   • Simulaciones históricas")
        print("   • Métricas de riesgo y retorno")
        print("   • Exportación a Excel")
        print("\n🌐 El dashboard se abrirá en tu navegador...")
        
        # Ejecutar Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado por el usuario")
    except Exception as e:
        print(f"❌ Error ejecutando el dashboard: {e}")
        print("💡 Asegúrate de que Streamlit esté instalado: pip install streamlit")

if __name__ == "__main__":
    main()