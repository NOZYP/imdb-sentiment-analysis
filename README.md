# 🎬 IMDB Sentiment Analysis with Rating Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-87.92%25-brightgreen)

**Análisis de sentimiento avanzado de reseñas de películas con predicción de rating en escala 1-10**

[Demo](#demo) • [Instalación](#instalación) • [Uso](#uso) • [API](#api) • [Resultados](#resultados)

</div>

---

## 📋 Descripción

Sistema completo de análisis de sentimiento para reseñas de películas del dataset IMDB que no solo clasifica opiniones como positivas o negativas, sino que también predice un **rating de 1 a 10 estrellas** basado en la intensidad del sentimiento.

### ✨ Características Principales

- 🎯 **Clasificación Binaria**: Sentimiento Positivo/Negativo con 87.92% de accuracy
- ⭐ **Predicción de Rating**: Escala 1-10 estrellas basada en intensidad de sentimiento
- 📊 **Análisis Detallado**: Métricas por rating individual y distribución de datos
- 🚀 **API REST**: Interfaz web interactiva y endpoints RESTful
- 📈 **Visualización**: Dashboard con estadísticas del modelo y distribución de datos
- 💾 **50K+ Reseñas**: Entrenado con 25,000 reseñas de entrenamiento y 25,000 de test

## 🏗️ Arquitectura del Proyecto

```
Proyecto/
├── model/                          # Módulo de entrenamiento
│   ├── modelo_sentiment.py        # Clase principal del modelo
│   └── requirements.txt            # Dependencias de entrenamiento
├── api/                            # API REST
│   ├── api_sentiment.py            # FastAPI application
│   ├── requirements.txt            # Dependencias de la API
│   └── README_API.md              # Documentación de la API
├── results/                        # Resultados y artefactos
│   ├── models/                    # Modelos entrenados
│   │   └── imdb_sentiment_model.pkl
│   ├── statistics/                # Estadísticas y métricas
│   │   └── model_statistics.json
│   └── predictions/               # Predicciones
│       └── unlabeled_predictions.csv
├── aclImdb/                       # Dataset IMDB (50k reviews)
│   ├── train/                     # 25k entrenamiento
│   │   ├── pos/                   # 12.5k positivas
│   │   ├── neg/                   # 12.5k negativas
│   │   └── unsup/                 # 50k sin etiquetar
│   └── test/                      # 25k prueba
│       ├── pos/                   # 12.5k positivas
│       └── neg/                   # 12.5k negativas
└── README.md                      # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 4GB RAM mínimo
- 2GB espacio en disco

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Instalar dependencias del modelo
cd model
pip install -r requirements.txt

# Instalar dependencias de la API (opcional)
cd ../api
pip install -r requirements.txt
```

### Dataset

El proyecto utiliza el [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) de Stanford.

## 📖 Uso

### 1. Entrenar el Modelo

```bash
cd model
python modelo_sentiment.py
```

Este proceso:
- ✅ Carga 50,000 reseñas del dataset IMDB
- ✅ Preprocesa texto (tokenización, lemmatización, stopwords)
- ✅ Entrena modelo de Logistic Regression con TF-IDF
- ✅ Evalúa con métricas detalladas
- ✅ Guarda modelo en `results/models/imdb_sentiment_model.pkl`
- ✅ Guarda estadísticas en `results/statistics/model_statistics.json`

**Tiempo estimado**: 15-20 minutos en hardware estándar

### 2. Usar el Modelo (Python)

```python
from modelo_sentiment import IMDBSentimentClassifier
import pickle

# Cargar modelo entrenado
classifier = IMDBSentimentClassifier('ruta/al/dataset')
with open('../results/models/imdb_sentiment_model.pkl', 'rb') as f:
    classifier.model = pickle.load(f)

# Predecir sentimiento y rating
review = "This movie was absolutely fantastic! Best film ever!"
sentiment, rating, proba = classifier.predict_sentiment_with_rating(review)

print(f"Sentimiento: {'Positive' if sentiment == 1 else 'Negative'}")
print(f"Rating: {rating}/10 estrellas")
print(f"Confianza: {max(proba):.2%}")
```

### 3. Ejecutar la API

```bash
cd api
python api_sentiment.py
```

Accede a:
- **Aplicación Web**: http://localhost:8001
- **Documentación API**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health

## 🌐 API

### Endpoints Principales

#### POST `/predict`
Predice el sentimiento y rating de una reseña.

**Request:**
```json
{
  "text": "Amazing movie! Loved every minute of it."
}
```

**Response:**
```json
{
  "text": "Amazing movie! Loved every minute of it.",
  "sentiment": "Positive",
  "sentiment_label": 1,
  "rating": 9,
  "confidence": 0.9234,
  "probabilities": {
    "negative": 0.0766,
    "positive": 0.9234
  }
}
```

#### GET `/statistics`
Obtiene las estadísticas completas del modelo.

**Response:**
```json
{
  "accuracy": 0.8792,
  "f1_score": 0.8801,
  "precision": 0.8788,
  "recall": 0.8862,
  "confusion_matrix": [[10903, 1597], [1422, 11078]],
  "rating_accuracy": {
    "1": {"accuracy": 0.9333, "count": 5022},
    "10": {"accuracy": 0.9226, "count": 4999}
  }
}
```

#### GET `/distribution`
Obtiene la distribución de datos de entrenamiento y test.

### Ejemplo con cURL

```bash
# Predecir sentimiento
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie, waste of time!"}'

# Obtener estadísticas
curl "http://localhost:8001/statistics"
```

### Ejemplo con Python

```python
import requests

# Predecir sentimiento
response = requests.post(
    "http://localhost:8001/predict",
    json={"text": "Brilliant film with amazing acting!"}
)
result = response.json()
print(f"Rating: {result['rating']}/10")
print(f"Sentiment: {result['sentiment']}")
```

## 📊 Resultados

### Métricas del Modelo

| Métrica | Valor |
|---------|-------|
| **Accuracy** | 87.92% |
| **F1 Score** | 0.8801 |
| **Precision** | 0.8788 |
| **Recall** | 0.8862 |

### Matriz de Confusión

|  | Predicho Neg | Predicho Pos |
|---|--------------|--------------|
| **Real Neg** | 10,903 | 1,597 |
| **Real Pos** | 1,422 | 11,078 |

### Accuracy por Rating

| Rating | Accuracy | Cantidad |
|--------|----------|----------|
| ⭐ 1 | 93.33% | 5,022 |
| ⭐⭐ 2 | 90.10% | 2,302 |
| ⭐⭐⭐ 3 | 86.11% | 2,541 |
| ⭐⭐⭐⭐ 4 | 74.16% | 2,635 |
| ⭐⭐⭐⭐⭐⭐⭐ 7 | 78.85% | 2,307 |
| ⭐⭐⭐⭐⭐⭐⭐⭐ 8 | 88.14% | 2,850 |
| ⭐⭐⭐⭐⭐⭐⭐⭐⭐ 9 | 91.08% | 2,344 |
| ⭐⭐⭐⭐⭐⭐⭐⭐⭐⭐ 10 | 92.26% | 4,999 |

### Distribución de Datos

**Conjunto de Entrenamiento (25,000 reseñas)**
- Positivas: 12,500 (rating promedio: 8.74 ± 1.16)
- Negativas: 12,500 (rating promedio: 2.22 ± 1.19)

**Conjunto de Test (25,000 reseñas)**
- Positivas: 12,500 (rating promedio: 8.74 ± 1.16)
- Negativas: 12,500 (rating promedio: 2.22 ± 1.19)

## 🧪 Ejemplos de Predicción

### Muy Positivo
```
Input: "Masterpiece! Best movie I've ever seen. Amazing acting and brilliant story."
Output: Positive | Rating: 10/10 | Confidence: 97.8%
```

### Muy Negativo
```
Input: "Worst movie ever made. Horrible in every way. Complete waste of time."
Output: Negative | Rating: 1/10 | Confidence: 99.3%
```

### Moderadamente Positivo
```
Input: "It was a good movie. I enjoyed it but it wasn't perfect."
Output: Positive | Rating: 7/10 | Confidence: 69.4%
```

### Moderadamente Negativo
```
Input: "Not very good. I was disappointed with this film."
Output: Negative | Rating: 3/10 | Confidence: 66.3%
```

## 🛠️ Tecnologías

- **Python 3.8+**: Lenguaje principal
- **Scikit-learn**: Machine Learning (Logistic Regression, TF-IDF)
- **NLTK**: Procesamiento de lenguaje natural
- **FastAPI**: Framework web para la API
- **Uvicorn**: Servidor ASGI
- **Pydantic**: Validación de datos
- **Pandas**: Análisis de datos (opcional)
- **NumPy**: Operaciones numéricas

## 📈 Metodología

### 1. Preprocesamiento
- Conversión a minúsculas
- Eliminación de HTML tags
- Eliminación de caracteres especiales
- Tokenización con NLTK
- Lemmatización
- Eliminación de stopwords
- Filtrado de palabras cortas (<3 caracteres)

### 2. Vectorización
- **TF-IDF Vectorizer**
  - Max features: 10,000
  - N-grams: (1, 2)
  - Min document frequency: 5
  - Max document frequency: 0.8

### 3. Modelo
- **Algoritmo**: Logistic Regression
- **Regularización**: L2 (C=0.5)
- **Solver**: lbfgs
- **Max iterations**: 1,000
- **Class weight**: Balanced

### 4. Predicción de Rating
El rating 1-10 se calcula usando la probabilidad de sentimiento positivo:
- **Probabilidad < 0.5**: Rating 1-4 (negativo)
- **Probabilidad ≥ 0.5**: Rating 7-10 (positivo)
- Mapeo basado en distribución real del dataset

## 🎨 Interfaz Web

La aplicación incluye una interfaz web moderna con tres secciones:

### 1. **Tab Predicción**
- Campo de texto para escribir reseñas
- Botones con ejemplos rápidos
- Resultado con sentimiento, rating con estrellas y confianza

### 2. **Tab Estadísticas**
- Métricas del modelo (Accuracy, F1, Precision, Recall)
- Matriz de confusión
- Datos del conjunto de prueba

### 3. **Tab Distribución**
- Distribución de ratings en entrenamiento y test
- Accuracy por rating individual
- Estadísticas descriptivas

## 🤝 Contribuir

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Dataset**: [Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/) por Andrew Maas et al. (Stanford University)
- **Paper**: [Learning Word Vectors for Sentiment Analysis](http://www.aclweb.org/anthology/P11-1015) (ACL 2011)

## 📧 Contacto

Para preguntas o sugerencias, abre un issue en GitHub o contacta al equipo de desarrollo.

---

<div align="center">

**⭐ Si te gusta este proyecto, dale una estrella en GitHub! ⭐**

Desarrollado con ❤️ para análisis de sentimiento avanzado

</div>
