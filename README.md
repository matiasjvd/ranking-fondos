# 📈 Dashboard de Análisis de Fondos

Un dashboard interactivo profesional para el análisis y ranking de fondos de inversión con métricas de performance avanzadas y optimización de portafolios.

## 🚀 Características

- **Análisis de Performance Completo**: YTD, MTD, retornos anuales, volatilidad, máximo drawdown, VaR y CVaR
- **Sistema de Scoring Personalizable**: Ranking basado en Z-scores con pesos configurables
- **Filtros Avanzados**: Por región, clase de activo, subclase y sector
- **Frontera Eficiente**: Análisis de optimización de portafolios con CVXPY
- **Visualizaciones Interactivas**: Gráficos de retornos acumulados y composición de portafolios
- **Exportación**: Reportes en CSV y PDF

## 📊 Datos

El dashboard utiliza dos archivos CSV principales:

- `data/funds_prices.csv`: Precios históricos de 329 fondos desde 2006
- `data/funds_dictionary.csv`: Metadata de los fondos (región, clase de activo, sector, etc.)

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd ranking-fondos
```

2. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

3. **Convertir datos (si es necesario)**:
Si tienes los archivos Excel originales, ejecuta:
```bash
python convert_data.py
```

4. **Ejecutar el dashboard**:
```bash
streamlit run funds_dashboard.py
```

## 📁 Estructura del Proyecto

```
ranking-fondos/
├── funds_dashboard.py      # Dashboard principal de Streamlit
├── convert_data.py         # Script para convertir Excel a CSV
├── requirements.txt        # Dependencias de Python
├── README.md              # Este archivo
└── data/                  # Directorio de datos
    ├── funds_prices.csv   # Precios históricos de fondos
    └── funds_dictionary.csv # Metadata de fondos
```

## 🎯 Uso del Dashboard

### 🌙 Tema Oscuro
- **Por defecto**: El dashboard inicia en modo oscuro para una experiencia visual elegante
- **Toggle disponible**: Cambia entre modo oscuro y claro desde el sidebar
- **Gráficos optimizados**: Colores y contrastes adaptados para cada tema

### Filtros
- **Región**: Filtra fondos por geografía (América del Norte, Europa, etc.)
- **Clase de Activo**: Equity, Fixed Income, Commodities, etc.
- **Subclase**: Categorías más específicas dentro de cada clase
- **Sector**: Sectores específicos (Technology, Healthcare, etc.)

### Análisis de Performance
- **Métricas Calculadas**: 
  - Retornos: YTD, MTD, mensual, 1 año, 2025, 2024, 2023
  - Riesgo: Volatilidad anualizada, máximo drawdown
  - VaR/CVaR: Value at Risk y Conditional VaR al 5%

### Scoring Personalizado
- Configura pesos para cada métrica (0-50%)
- El sistema calcula Z-scores normalizados
- Ranking automático basado en el score compuesto

### Frontera Eficiente
- Selecciona 2-10 fondos para análisis
- Optimización de portafolios usando programación cuadrática
- Visualización de la frontera eficiente
- Identificación del portafolio óptimo (máximo ratio de Sharpe)

### Exportación
- **CSV**: Datos completos de performance y rankings
- **PDF**: Reporte ejecutivo con top fondos y metodología

## 🔧 Dependencias Principales

- **Streamlit**: Framework web para el dashboard
- **Pandas/NumPy**: Procesamiento de datos
- **Plotly**: Visualizaciones interactivas
- **CVXPY**: Optimización de portafolios
- **ReportLab**: Generación de reportes PDF

## 📈 Métricas de Performance

### Retornos
- **YTD**: Año corriente hasta la fecha
- **MTD**: Mes corriente hasta la fecha
- **1Y**: Retorno de 12 meses
- **Anuales**: 2025, 2024, 2023

### Métricas de Riesgo
- **Volatilidad**: Desviación estándar anualizada
- **Max Drawdown**: Máxima caída desde un pico
- **VaR 5%**: Value at Risk al 95% de confianza
- **CVaR 5%**: Expected Shortfall (promedio del peor 5%)

### Scoring
- **Z-Score**: Normalización estadística de métricas
- **Pesos Configurables**: Personalización del ranking
- **Dirección de Métricas**: Positivas (mayor es mejor) vs negativas (menor es mejor)

## 🚀 Características Avanzadas

### Optimización de Portafolios
- Implementación de la teoría moderna de portafolios
- Restricciones: solo posiciones largas, suma de pesos = 100%
- Objetivo: minimizar riesgo para un retorno dado
- Identificación del portafolio con máximo ratio de Sharpe

### Filtros Cascada
- Los filtros se actualizan dinámicamente
- Selección de región afecta clases de activo disponibles
- Lógica intuitiva para navegación de datos

### Visualizaciones Profesionales
- Gráficos de retornos acumulados con base 100
- Matriz de correlación para análisis de diversificación
- Gráficos de composición de portafolios
- Frontera eficiente interactiva

## 🔍 Troubleshooting

### Error de archivos no encontrados
```bash
# Ejecutar el script de conversión
python convert_data.py
```

### Problemas de optimización
- Verificar que hay suficientes datos históricos
- Asegurar que los fondos seleccionados tienen datos completos
- Revisar el debug container para detalles específicos

### Performance lenta
- Reducir el número de fondos analizados
- Usar filtros para limitar el dataset
- El caching de Streamlit optimiza cálculos repetidos

## 📝 Notas Técnicas

- Los datos se cargan automáticamente desde CSV
- Caching inteligente para optimizar performance
- Manejo robusto de errores y datos faltantes
- Interfaz responsive para diferentes tamaños de pantalla

## 🤝 Contribuciones

Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.