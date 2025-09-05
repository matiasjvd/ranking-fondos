# 📊 Comparación de Dashboards

## 🔄 Opciones Disponibles

### **Dashboard Original** (`funds_dashboard.py`)
```bash
streamlit run funds_dashboard.py
```
- ✅ **Funcionalidad completa** de análisis de fondos
- ✅ **Código original intacto** y sin modificaciones
- ✅ **Todas las características** existentes funcionando
- ✅ **Filtros avanzados** y métricas de performance
- ✅ **Gráficos interactivos** de retornos acumulados
- ✅ **Frontera eficiente** y optimización
- ✅ **Exportación** a PDF y Excel

### **Dashboard con Portafolios** (`dashboard_with_portfolio.py`)
```bash
python start_portfolio_dashboard.py
# o
streamlit run dashboard_with_portfolio.py --server.port 8502
```
- ✅ **Todo lo del dashboard original** (funciones reutilizadas)
- 🆕 **Constructor de portafolios** interactivo
- 🆕 **Carrito de fondos** en tiempo real
- 🆕 **Gestión de pesos** y categorías
- 🆕 **Métricas de portafolio** (Sharpe, Drawdown, etc.)
- 🆕 **Simulaciones históricas** personalizables
- 🆕 **Exportación de portafolios** a Excel

## 📋 Tabla Comparativa

| Característica | Dashboard Original | Dashboard + Portafolios |
|---|---|---|
| **Análisis de fondos individuales** | ✅ Completo | ✅ Completo |
| **Filtros y búsqueda** | ✅ Avanzados | ✅ Básicos |
| **Métricas de performance** | ✅ Extensas | ✅ Básicas |
| **Gráficos de retornos** | ✅ Múltiples fondos | ✅ Múltiples fondos |
| **Frontera eficiente** | ✅ Sí | ❌ No |
| **Scoring personalizado** | ✅ Sí | ❌ No |
| **Exportación PDF** | ✅ Sí | ❌ No |
| **Constructor de portafolios** | ❌ No | ✅ **Sí** |
| **Carrito de fondos** | ❌ No | ✅ **Sí** |
| **Gestión de pesos** | ❌ No | ✅ **Sí** |
| **Métricas de portafolio** | ❌ No | ✅ **Sí** |
| **Simulaciones históricas** | ❌ No | ✅ **Sí** |
| **Exportación de portafolios** | ❌ No | ✅ **Sí** |

## 🎯 Casos de Uso Recomendados

### **Usar Dashboard Original cuando:**
- 🔍 Necesitas **análisis detallado** de fondos individuales
- 📊 Quieres **todas las métricas** disponibles (VaR, CVaR, etc.)
- 🎯 Necesitas **frontera eficiente** y optimización
- 📑 Requieres **reportes en PDF** profesionales
- ⚖️ Quieres **scoring personalizado** con pesos ajustables
- 🔬 Realizas **análisis técnico** profundo

### **Usar Dashboard con Portafolios cuando:**
- 🛒 Quieres **construir portafolios** de manera interactiva
- 💼 Necesitas **gestionar múltiples activos** con pesos específicos
- 📈 Quieres **simular performance** de tu portafolio
- 💾 Necesitas **exportar composiciones** de portafolio
- 🎨 Prefieres una **interfaz más simple** y directa
- 🔄 Quieres **experimentar** con diferentes combinaciones

## 🚀 Recomendación de Uso

### **Flujo de Trabajo Sugerido:**

1. **Fase de Investigación**: Usa el **Dashboard Original**
   - Analiza fondos individuales en detalle
   - Aplica filtros avanzados
   - Revisa métricas completas
   - Identifica candidatos para tu portafolio

2. **Fase de Construcción**: Cambia al **Dashboard con Portafolios**
   - Agrega los fondos seleccionados al carrito
   - Ajusta pesos según tu estrategia
   - Simula performance histórica
   - Exporta la composición final

3. **Fase de Monitoreo**: Alterna entre ambos
   - Usa el original para análisis periódicos
   - Usa el de portafolios para ajustes de asignación

## 🔧 Instalación y Configuración

### **Requisitos Comunes**
```bash
pip install streamlit pandas plotly numpy cvxpy xlsxwriter reportlab
```

### **Datos Necesarios**
```bash
# Generar archivos CSV (solo una vez)
python convert_data.py
```

### **Ejecución Simultánea**
Puedes ejecutar ambos dashboards al mismo tiempo en puertos diferentes:

```bash
# Terminal 1: Dashboard Original
streamlit run funds_dashboard.py --server.port 8501

# Terminal 2: Dashboard con Portafolios  
streamlit run dashboard_with_portfolio.py --server.port 8502
```

## 📁 Estructura de Archivos

```
ranking-fondos/
├── funds_dashboard.py              # 📊 Dashboard original (INTACTO)
├── dashboard_with_portfolio.py     # 🛒 Dashboard con portafolios
├── portfolio_builder.py            # 🔧 Módulo de portafolios (no usado)
├── main_dashboard.py              # 🔄 Versión con pestañas (alternativa)
├── start_portfolio_dashboard.py   # 🚀 Script de inicio
├── convert_data.py                # 📁 Conversor de datos
├── data/
│   ├── funds_prices.csv           # 💰 Precios históricos
│   └── funds_dictionary.csv       # 📖 Metadatos de fondos
└── README files...                # 📚 Documentación
```

## 🎯 Conclusión

**Objetivo Cumplido**: Se ha implementado exitosamente la funcionalidad de constructor de portafolios como un **módulo adicional** que:

✅ **Mantiene intacto** el proyecto original  
✅ **Agrega funcionalidad** de portafolios  
✅ **No modifica** el código existente  
✅ **Proporciona interfaz** intuitiva tipo carrito  
✅ **Incluye simulaciones** y métricas  
✅ **Permite exportación** de resultados  

Ambas versiones coexisten perfectamente y pueden usarse según las necesidades específicas del análisis.