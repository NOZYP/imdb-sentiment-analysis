# Modelos Entrenados

Esta carpeta contiene los modelos entrenados del sistema de análisis de sentimiento.

## Archivos

### `imdb_sentiment_model.pkl`
- **Tipo**: Modelo de Scikit-learn serializado con pickle
- **Tamaño**: ~45 MB
- **Contenido**: Pipeline completo (TF-IDF + Logistic Regression)
- **Generado por**: `model/modelo_sentiment.py`

## Uso

```python
import pickle
from modelo_sentiment import IMDBSentimentClassifier

# Cargar modelo
classifier = IMDBSentimentClassifier(data_path)
with open('results/models/imdb_sentiment_model.pkl', 'rb') as f:
    classifier.model = pickle.load(f)

# Usar modelo
sentiment, rating, proba = classifier.predict_sentiment_with_rating("Great movie!")
```

## Notas

- Este archivo se regenera cada vez que se ejecuta el entrenamiento
- No se incluye en el repositorio por su tamaño (ver `.gitignore`)
- Para obtener el modelo, ejecuta `python model/modelo_sentiment.py`
