#!/usr/bin/env python3
"""
DASHBOARD WITH CART LAUNCHER
Script para ejecutar el dashboard original con carrito de portafolios integrado
"""

import subprocess
import sys
import os

def main():
    """Ejecutar el dashboard con carrito integrado"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(script_dir, 'dashboard_with_cart.py')
        
        print("🚀 Iniciando Dashboard de Fondos con Carrito de Portafolios...")
        print("\n✨ Funcionalidades:")
        print("   📊 Análisis completo de fondos (funcionalidad original)")
        print("   🛒 Carrito de portafolios integrado en el sidebar")
        print("   ➕ Botones 'Agregar' en cada fondo")
        print("   ⚖️ Gestión de pesos y categorías")
        print("   📈 Análisis de performance del portafolio")
        print("   💾 Exportación a Excel")
        print("\n🌐 Abriendo en el navegador...")
        print("📍 URL: http://localhost:8503")
        
        # Ejecutar Streamlit en puerto específico
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8503",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que Streamlit esté instalado: pip install streamlit")

if __name__ == "__main__":
    main()