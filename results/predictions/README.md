# Predicciones

Esta carpeta contiene las predicciones realizadas sobre datos sin etiquetar.

## Archivos

### `unlabeled_predictions.csv`
- **Tipo**: CSV
- **Tamaño**: ~50 MB (50,000 reseñas)
- **Contenido**: Predicciones sobre el conjunto `aclImdb/train/unsup/`
- **Generado por**: `model/modelo_sentiment.py` (opcional)

## Estructura del CSV

```csv
review,sentiment,rating,confidence
"Great movie! Loved it...",Positive,9,0.9234
"Terrible waste of time...",Negative,1,0.9876
...
```

## Columnas

- **review**: Texto de la reseña (truncado a 200 caracteres)
- **sentiment**: Sentimiento predicho (Positive/Negative)
- **rating**: Rating predicho (1-10 estrellas)
- **confidence**: Confianza de la predicción (0-1)

## Uso

```python
import pandas as pd

# Cargar predicciones
df = pd.read_csv('results/predictions/unlabeled_predictions.csv')

# Analizar distribución
print(df['sentiment'].value_counts())
print(df['rating'].value_counts().sort_index())
print(f"Confianza promedio: {df['confidence'].mean():.4f}")
```

## Generación

Para generar este archivo, ejecuta el entrenamiento y cuando se pregunte:

```
Do you want to process the unlabeled data now? (y/n): y
```

O desde Python:

```python
classifier.process_unlabeled_data()
```

## Notas

- Este archivo es opcional y puede no existir
- No se incluye en el repositorio por su tamaño
- Útil para análisis de datos no etiquetados
- Toma ~5-10 minutos en generar
