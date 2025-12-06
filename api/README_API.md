# IMDB Sentiment Analysis API

API web interactiva para análisis de sentimiento de reseñas de películas con FastAPI.

## Características

- 🎯 **Predicción de Sentimiento**: Analiza reseñas y predice si son positivas o negativas
- ⭐ **Rating 1-10**: Genera un rating de estrellas basado en la intensidad del sentimiento
- 📊 **Estadísticas del Modelo**: Visualiza accuracy, F1, precisión, recall y matriz de confusión
- 📈 **Distribución de Datos**: Explora cómo se distribuyen los ratings en el dataset
- 🎨 **Interfaz Interactiva**: UI moderna y fácil de usar

## Instalación

1. Asegúrate de tener el modelo entrenado (`imdb_sentiment_model.pkl`)

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python api_sentiment.py
```

O usando uvicorn directamente:
```bash
uvicorn api_sentiment:app --reload --host 0.0.0.0 --port 8000
```

## Acceso

- **Aplicación Web**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Endpoints

### GET /
Página principal con interfaz interactiva (HTML)

### POST /predict
Predice el sentimiento de una reseña

**Request Body:**
```json
{
  "text": "This movie was fantastic!"
}
```

**Response:**
```json
{
  "text": "This movie was fantastic!",
  "sentiment": "Positive",
  "sentiment_label": 1,
  "rating": 9,
  "confidence": 0.9123,
  "probabilities": {
    "negative": 0.0877,
    "positive": 0.9123
  }
}
```

### GET /statistics
Obtiene estadísticas del modelo (accuracy, F1, confusion matrix, etc.)

### GET /distribution
Obtiene la distribución de ratings en el dataset

### GET /health
Health check del servicio

## Uso desde Python

```python
import requests

# Predecir sentimiento
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Amazing movie, loved every minute!"}
)
result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Rating: {result['rating']}/10")
```

## Características de la Interfaz

### Tab 1: Predicción
- Campo de texto para escribir reseñas
- Botones con ejemplos rápidos (positivo, negativo, neutral)
- Muestra sentimiento, rating con estrellas, confianza y probabilidades

### Tab 2: Estadísticas
- Métricas del modelo: Accuracy, F1 Score, Precisión, Recall
- Matriz de confusión visualizada
- Datos del conjunto de prueba

### Tab 3: Distribución
- Distribución de ratings en entrenamiento (promedio, std, rango)
- Accuracy por rating individual (1-10)
- Conteo de muestras por categoría

## Notas

- Las estadísticas se generan la primera vez que se accede a ellas y se cachean en `model_statistics.json`
- El modelo debe estar entrenado antes de ejecutar la API
- La generación inicial de estadísticas puede tomar varios minutos
