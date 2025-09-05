#!/usr/bin/env python3
"""
SIMPLE CART DASHBOARD LAUNCHER
Script para ejecutar el dashboard original con carrito simple (checkboxes)
"""

import subprocess
import sys
import os

def main():
    """Ejecutar el dashboard con carrito simple"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(script_dir, 'dashboard_original_with_simple_cart.py')
        
        print("🚀 Iniciando Dashboard Original + Carrito Simple...")
        print("\n✨ Funcionalidades:")
        print("   📊 Dashboard original COMPLETO (sin modificaciones)")
        print("   ✅ Todas las métricas y análisis originales")
        print("   ✅ Frontera eficiente completa")
        print("   ✅ Scoring personalizado")
        print("   🛒 Carrito simple con checkboxes")
        print("   📈 Análisis completo del carrito en pestaña separada")
        print("   ⚖️ Gestión de pesos por activo")
        print("   🎯 Frontera eficiente del carrito")
        print("   💾 Exportación a Excel")
        print("\n🌐 Abriendo en el navegador...")
        print("📍 URL: http://localhost:8504")
        
        # Ejecutar Streamlit en puerto específico
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8504",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard cerrado")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de que Streamlit esté instalado: pip install streamlit")

if __name__ == "__main__":
    main()