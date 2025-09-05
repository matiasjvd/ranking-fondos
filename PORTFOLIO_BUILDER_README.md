# 🛒 Constructor de Portafolios - Documentación

## 📋 Descripción General

El Constructor de Portafolios es un módulo adicional que extiende el Dashboard de Análisis de Fondos existente sin modificar su funcionalidad original. Permite a los usuarios construir, analizar y gestionar portafolios personalizados de fondos de inversión.

## 🚀 Características Principales

### 1. **Selección y Gestión de Activos**
- ➕ Agregar fondos al portafolio desde la tabla de análisis principal
- 🛒 Widget de "carrito" en el sidebar que muestra el estado del portafolio
- 🗑️ Remover activos individuales del portafolio
- 🧹 Limpiar todo el portafolio de una vez

### 2. **Configuración de Pesos y Categorías**
- ⚖️ Ajustar pesos individuales de cada activo (0-100%)
- 🔄 Normalización automática para que los pesos sumen 100%
- 🏷️ Categorización de activos (Renta Fija, Renta Variable, Alternativos, etc.)
- 📊 Visualización de asignación por categorías

### 3. **Análisis de Performance**
- 📈 Gráfico de performance histórica del portafolio
- 📊 Cálculo de métricas de riesgo y retorno:
  - Retorno total del período
  - Volatilidad anualizada
  - Sharpe Ratio
  - Maximum Drawdown
  - VaR y CVaR al 5%
- 📅 Análisis personalizable por períodos de tiempo

### 4. **Visualizaciones Interactivas**
- 🥧 Gráfico de pie para asignación de activos
- 📈 Gráfico de evolución temporal del portafolio
- 📊 Métricas en tiempo real
- 🎨 Tema oscuro optimizado

### 5. **Exportación y Descarga**
- 📊 Exportación completa a Excel con múltiples hojas:
  - Composición del portafolio
  - Métricas calculadas
  - Asignación por categorías
- 📁 Nombres de archivo con timestamp automático

## 🛠️ Instalación y Uso

### Requisitos Previos
El módulo utiliza las mismas dependencias que el dashboard original:
```bash
pip install streamlit pandas plotly numpy cvxpy xlsxwriter
```

### Ejecución

#### Opción 1: Dashboard Completo (Recomendado)
```bash
python run_with_portfolio.py
```

#### Opción 2: Streamlit Directo
```bash
streamlit run main_dashboard.py
```

#### Opción 3: Dashboard Original (Sin Portafolios)
```bash
streamlit run funds_dashboard.py
```

## 📖 Guía de Uso

### 1. **Agregar Fondos al Portafolio**
1. Ve a la pestaña "📊 Análisis de Fondos"
2. Busca y filtra los fondos de tu interés
3. Haz clic en "➕ Agregar" junto al fondo deseado
4. El fondo aparecerá en el widget del carrito en el sidebar

### 2. **Gestionar el Portafolio**
1. Ve a la pestaña "🛒 Constructor de Portafolios"
2. Ajusta los pesos de cada activo usando los controles numéricos
3. Asigna categorías a cada fondo
4. Usa "⚖️ Normalizar a 100%" para ajustar automáticamente los pesos

### 3. **Analizar Performance**
1. En la pestaña de portafolios, selecciona el período de análisis
2. Haz clic en "🔄 Calcular Métricas"
3. Revisa las métricas calculadas y el gráfico de performance
4. Descarga los resultados usando "📊 Descargar Portafolio (Excel)"

## 🏗️ Arquitectura Técnica

### Estructura de Archivos
```
ranking-fondos/
├── funds_dashboard.py          # Dashboard original (sin modificar)
├── portfolio_builder.py        # Módulo del constructor de portafolios
├── main_dashboard.py          # Dashboard integrado con pestañas
├── run_with_portfolio.py      # Script de ejecución
└── PORTFOLIO_BUILDER_README.md # Esta documentación
```

### Componentes Principales

#### `PortfolioBuilder` (Clase Principal)
- Gestión del estado del portafolio en `st.session_state`
- Cálculo de métricas de riesgo y retorno
- Generación de visualizaciones
- Exportación de datos

#### Funciones de Renderizado
- `render_portfolio_cart_widget()`: Widget del carrito en sidebar
- `render_add_to_portfolio_button()`: Botones para agregar fondos
- `render_portfolio_management_tab()`: Pestaña completa de gestión

#### Integración No Intrusiva
- El dashboard original permanece completamente intacto
- Uso de pestañas para separar funcionalidades
- Estado compartido a través de `st.session_state`

## 📊 Métricas Calculadas

### Retorno y Riesgo
- **Retorno Total**: Retorno acumulado del portafolio en el período
- **Volatilidad Anualizada**: Desviación estándar de retornos anualizados
- **Sharpe Ratio**: Retorno ajustado por riesgo (asumiendo risk-free rate = 0)

### Métricas de Riesgo
- **Maximum Drawdown**: Máxima pérdida desde un pico histórico
- **VaR 5%**: Value at Risk al 5% de confianza (anualizado)
- **CVaR 5%**: Conditional VaR - pérdida esperada en el peor 5% de casos

## 🔧 Personalización

### Agregar Nuevas Categorías
Modifica la lista en `render_portfolio_management_tab()`:
```python
options=["Renta Fija", "Renta Variable", "Alternativos", "Mixto", "Nueva Categoría"]
```

### Modificar Métricas
Extiende el método `calculate_portfolio_metrics()` en la clase `PortfolioBuilder`.

### Personalizar Visualizaciones
Los gráficos utilizan Plotly y pueden modificarse en:
- `create_portfolio_performance_chart()`
- `create_allocation_chart()`

## 🚨 Limitaciones y Consideraciones

1. **Datos Históricos**: Las métricas dependen de la disponibilidad de datos históricos
2. **Performance**: El cálculo de métricas puede ser lento con muchos activos
3. **Memoria**: El estado del portafolio se mantiene en sesión (se pierde al cerrar)
4. **Validación**: Los pesos pueden no sumar exactamente 100% debido a redondeo

## 🔮 Futuras Mejoras

- 💾 Persistencia de portafolios (guardar/cargar)
- 🎯 Optimización automática de portafolios
- 📈 Backtesting más avanzado
- 🔔 Alertas y notificaciones
- 📱 Versión móvil optimizada

## 🆘 Soporte y Troubleshooting

### Problemas Comunes

**Error: "No se pudieron calcular métricas"**
- Verifica que los fondos tengan datos en el período seleccionado
- Reduce el período de análisis

**Los pesos no suman 100%**
- Usa el botón "⚖️ Normalizar a 100%"
- Verifica que no hay valores negativos

**El portafolio se vacía al recargar**
- Esto es normal - el estado se mantiene solo durante la sesión
- Descarga el portafolio antes de cerrar

### Logs y Debug
El módulo incluye manejo de errores con mensajes informativos en la interfaz de Streamlit.

---

**Versión**: 1.0.0  
**Última actualización**: Septiembre 2024  
**Compatibilidad**: Python 3.8+, Streamlit 1.28+