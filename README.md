# House Prices Prediction Project

## Descripción
Proyecto de Machine Learning para predecir precios de viviendas usando técnicas avanzadas de preprocesamiento de datos y modelos predictivos.

## Estructura del Proyecto
```
ML-house-prices/
├── data/
│   ├── raw/           # Datos sin procesar (train.csv, test.csv)
│   └── processed/     # Datos procesados
├── notebooks/
│   ├── eda_01.ipynb           # Análisis exploratorio y tratamiento de datos faltantes
│   └── features_engineering.ipynb  # Ingeniería de características
├── functions/
│   ├── __init__.py
│   └── transformers.py        # Funciones de transformación personalizadas
└── README.md
```

## Procesamiento de Datos
### Manejo de Datos Faltantes
- Eliminación de features con >85% de datos faltantes
- Imputación específica para características del sótano
- Tratamiento especial para características del garaje
- Imputación basada en relaciones entre variables

### Features Engineering
- Transformadores personalizados para características del sótano
- Pipeline de transformación de datos
- Validación de transformaciones

## Visualizaciones
- Distribuciones de características del sótano
- Análisis de área y calidad del garaje
- Relaciones entre variables categóricas y numéricas

## Tecnologías Utilizadas
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Próximos Pasos
1. Completar pipeline de preprocesamiento
2. Implementar y evaluar modelos de ML
3. Optimización de hiperparámetros
4. Validación cruzada
5. Implementación en producción

## Contribución
Para contribuir al proyecto:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crea un Pull Request

## Licencia
Este proyecto está bajo la licencia MIT.