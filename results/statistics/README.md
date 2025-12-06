# Estadísticas del Modelo

Esta carpeta contiene las estadísticas y métricas del modelo entrenado.

## Archivos

### `model_statistics.json`
- **Tipo**: JSON
- **Tamaño**: ~10 KB
- **Contenido**: Métricas completas del modelo
- **Generado por**: `model/modelo_sentiment.py` durante la evaluación

## Estructura

```json
{
  "accuracy": 0.8792,
  "f1_score": 0.8801,
  "precision": 0.8788,
  "recall": 0.8862,
  "confusion_matrix": [[10903, 1597], [1422, 11078]],
  "rating_accuracy": {
    "1": {"accuracy": 0.9333, "count": 5022},
    ...
  },
  "training_distribution": {
    "Positive": {...},
    "Negative": {...}
  },
  "test_distribution": {
    "Positive": {...},
    "Negative": {...}
  }
}
```

## Uso

```python
import json

# Cargar estadísticas
with open('results/statistics/model_statistics.json', 'r') as f:
    stats = json.load(f)

print(f"Accuracy: {stats['accuracy']:.2%}")
print(f"F1 Score: {stats['f1_score']:.4f}")
```

## Notas

- Este archivo se regenera cada vez que se evalúa el modelo
- La API carga estas estadísticas automáticamente
- Se incluye en el repositorio para referencia rápida
