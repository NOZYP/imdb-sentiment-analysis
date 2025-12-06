# 🚀 Quick Start Guide

Guía rápida para comenzar a usar el sistema de análisis de sentimiento IMDB en 5 minutos.

## Paso 1: Requisitos Previos (1 min)

```bash
# Verificar Python 3.8+
python --version

# Verificar pip
pip --version
```

## Paso 2: Clonar e Instalar (2 min)

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Instalar dependencias del modelo
cd model
pip install -r requirements.txt
```

## Paso 3: Descargar Dataset (30 seg)

```bash
# Regresar al directorio raíz
cd ..

# Descargar dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# O descarga manualmente desde:
# http://ai.stanford.edu/~amaas/data/sentiment/
```

## Paso 4: Entrenar Modelo (15-20 min)

```bash
cd model
python modelo_sentiment.py
```

**Salida esperada:**
```
=== Loading Dataset ===
Loading 12500 reviews from ...
...
✓ Model meets accuracy requirements! (87.92%)
Model saved to ../results/models/imdb_sentiment_model.pkl
Statistics saved to ../results/statistics/model_statistics.json
```

## Paso 5: Ejecutar API (1 min)

```bash
# Instalar dependencias de API
cd ../api
pip install -r requirements.txt

# Ejecutar API
python api_sentiment.py
```

**Salida esperada:**
```
✓ Model loaded successfully
Access the application at: http://localhost:8001
```

## Paso 6: ¡Usar la Aplicación! (30 seg)

Abre tu navegador en: **http://localhost:8001**

### Prueba rápida:

1. **Tab Predicción**: Escribe una reseña de película
2. **Tab Estadísticas**: Ve las métricas del modelo
3. **Tab Distribución**: Explora la distribución de datos

## 🎯 Ejemplos de Uso

### Desde Python

```python
from modelo_sentiment import IMDBSentimentClassifier
import pickle

# Cargar modelo
classifier = IMDBSentimentClassifier('ruta/al/aclImdb')
with open('../results/models/imdb_sentiment_model.pkl', 'rb') as f:
    classifier.model = pickle.load(f)

# Predecir
review = "Amazing movie! Loved it!"
sentiment, rating, proba = classifier.predict_sentiment_with_rating(review)

print(f"Sentimiento: {'Positive' if sentiment == 1 else 'Negative'}")
print(f"Rating: {rating}/10")
print(f"Confianza: {max(proba):.2%}")
```

### Desde la API (cURL)

```bash
# Predecir sentimiento
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Great film! Highly recommend it."}'

# Ver estadísticas
curl "http://localhost:8001/statistics"
```

### Desde la API (Python requests)

```python
import requests

response = requests.post(
    "http://localhost:8001/predict",
    json={"text": "Worst movie ever. Don't waste your time."}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Rating: {result['rating']}/10")
```

## 📊 Resultados Esperados

Después del entrenamiento deberías ver:

- ✅ **Accuracy**: ~87.92%
- ✅ **F1 Score**: ~0.88
- ✅ **Archivos generados**:
  - `results/models/imdb_sentiment_model.pkl` (~45 MB)
  - `results/statistics/model_statistics.json` (~10 KB)

## 🔧 Troubleshooting

### Error: "Model file not found"
```bash
# Solución: Entrena el modelo primero
cd model
python modelo_sentiment.py
```

### Error: "No module named 'modelo_sentiment'"
```bash
# Solución: Ejecuta desde el directorio correcto
cd model  # Para entrenar
cd api    # Para la API
```

### Error: "Dataset not found"
```bash
# Solución: Descarga el dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

### Error: "Port 8001 already in use"
```bash
# Solución: Usa otro puerto
python api_sentiment.py --port 8002
# O mata el proceso existente
```

## 📚 Recursos Adicionales

- **Documentación completa**: Ver [README.md](README.md)
- **Estructura del proyecto**: Ver [STRUCTURE.md](STRUCTURE.md)
- **Documentación API**: Ver [api/README_API.md](api/README_API.md)

## ⏱️ Tiempos Estimados

| Actividad | Tiempo |
|-----------|--------|
| Instalación | 2-3 min |
| Descarga dataset | 2-5 min |
| Entrenamiento | 15-20 min |
| Configuración API | 1-2 min |
| **Total** | **20-30 min** |

## 🎉 ¡Listo!

Ahora tienes un sistema completo de análisis de sentimiento funcionando.

**Próximos pasos:**
- Explora la interfaz web
- Prueba con tus propias reseñas
- Consulta la documentación para usos avanzados
- Contribuye al proyecto en GitHub

---

**¿Problemas?** Abre un issue en GitHub o revisa la documentación completa.
