# Estructura del Proyecto

## Vista General

```
imdb-sentiment-analysis/
│
├── model/                          # 📊 Módulo de Entrenamiento
│   ├── modelo_sentiment.py        # Clase principal y entrenamiento
│   └── requirements.txt            # Dependencias del modelo
│
├── api/                            # 🌐 API REST
│   ├── api_sentiment.py            # FastAPI application
│   ├── requirements.txt            # Dependencias de la API
│   └── README_API.md              # Documentación de la API
│
├── results/                        # 💾 Resultados y Artefactos
│   ├── models/                    # Modelos entrenados
│   │   ├── imdb_sentiment_model.pkl
│   │   └── README.md
│   ├── statistics/                # Métricas y estadísticas
│   │   ├── model_statistics.json
│   │   └── README.md
│   └── predictions/               # Predicciones generadas
│       ├── unlabeled_predictions.csv
│       └── README.md
│
├── aclImdb/                       # 📁 Dataset (no incluido en repo)
│   ├── train/
│   │   ├── pos/                   # 12.5k reseñas positivas
│   │   ├── neg/                   # 12.5k reseñas negativas
│   │   └── unsup/                 # 50k sin etiquetar
│   └── test/
│       ├── pos/                   # 12.5k reseñas positivas
│       └── neg/                   # 12.5k reseñas negativas
│
├── README.md                      # 📖 Documentación principal
├── LICENSE                        # ⚖️ Licencia MIT
├── .gitignore                     # 🚫 Archivos ignorados por Git
└── STRUCTURE.md                   # 📋 Este archivo
```

## Descripción de Directorios

### `/model` - Módulo de Entrenamiento

Contiene todo el código relacionado con el entrenamiento del modelo.

**Archivos principales:**
- `modelo_sentiment.py`: Clase `IMDBSentimentClassifier` con todos los métodos
- `requirements.txt`: Dependencias necesarias para entrenamiento

**Responsabilidades:**
- Carga y preprocesamiento de datos
- Entrenamiento del modelo
- Evaluación y generación de métricas
- Guardado de modelo y estadísticas

**Ejecución:**
```bash
cd model
python modelo_sentiment.py
```

### `/api` - API REST

Contiene la aplicación FastAPI para servir el modelo.

**Archivos principales:**
- `api_sentiment.py`: Aplicación FastAPI con endpoints
- `requirements.txt`: Dependencias de la API
- `README_API.md`: Documentación detallada de la API

**Responsabilidades:**
- Carga del modelo entrenado
- Endpoints REST para predicciones
- Interfaz web interactiva
- Servir estadísticas del modelo

**Ejecución:**
```bash
cd api
python api_sentiment.py
```

**Acceso:**
- Web UI: http://localhost:8001
- API Docs: http://localhost:8001/docs

### `/results` - Resultados

Almacena todos los artefactos generados.

#### `/results/models`
- Modelos entrenados en formato pickle
- ~45 MB por modelo
- Ignorado en Git (demasiado grande)

#### `/results/statistics`
- Métricas en formato JSON
- ~10 KB
- Incluido en Git para referencia

#### `/results/predictions`
- Predicciones sobre datos sin etiquetar
- ~50 MB (50,000 reseñas)
- Ignorado en Git (archivo grande, opcional)

### `/aclImdb` - Dataset

Dataset IMDB Large Movie Review.

**Características:**
- 50,000 reseñas etiquetadas (25k train, 25k test)
- 50,000 reseñas sin etiquetar
- Formato: archivos .txt con nombre `[id]_[rating].txt`
- Ignorado en Git (750 MB)

**Descarga:**
```bash
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

## Flujo de Trabajo

### 1. Instalación Inicial

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis

# Descargar dataset (si no lo tienes)
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz

# Instalar dependencias del modelo
cd model
pip install -r requirements.txt
```

### 2. Entrenar Modelo

```bash
cd model
python modelo_sentiment.py
```

**Genera:**
- `results/models/imdb_sentiment_model.pkl`
- `results/statistics/model_statistics.json`

### 3. Ejecutar API

```bash
cd api
pip install -r requirements.txt
python api_sentiment.py
```

**Requiere:**
- `results/models/imdb_sentiment_model.pkl` (generado en paso 2)
- `results/statistics/model_statistics.json` (generado en paso 2)

### 4. Usar la Aplicación

Abre tu navegador en http://localhost:8001

## Dependencias

### Modelo (Training)
```
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
nltk>=3.8.0
```

### API (Deployment)
```
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
scikit-learn>=1.3.0
numpy>=1.24.0
nltk>=3.8.0
```

## Archivos de Configuración

### `.gitignore`
Define qué archivos no se suben a Git:
- Dataset completo (`aclImdb/`)
- Modelos entrenados (`.pkl`)
- Archivos Python compilados (`__pycache__/`)
- Archivos de entorno (`.env`, `venv/`)

### `LICENSE`
Licencia MIT - uso libre con atribución

### `README.md`
Documentación principal del proyecto

## Tamaños de Archivos

| Archivo/Directorio | Tamaño | Incluido en Git |
|-------------------|--------|-----------------|
| Dataset (`aclImdb/`) | ~750 MB | ❌ No |
| Modelo (`.pkl`) | ~45 MB | ❌ No |
| Estadísticas (`.json`) | ~10 KB | ✅ Sí |
| Predicciones (`.csv`) | ~50 MB | ❌ No |
| Código fuente | ~100 KB | ✅ Sí |
| Total repositorio | ~110 KB | - |
| Total con archivos generados | ~845 MB | - |

## Notas Importantes

1. **Dataset**: Debe descargarse manualmente (no incluido en repo)
2. **Modelo**: Se genera al entrenar (no incluido en repo)
3. **Rutas**: El código usa rutas relativas desde su directorio
4. **Python Path**: La API ajusta sys.path para importar desde `model/`
5. **NLTK Data**: Se descarga automáticamente al ejecutar

## Comandos Útiles

```bash
# Ver estructura del proyecto
tree /F  # Windows
tree -L 3  # Linux/Mac

# Verificar imports del modelo
cd model && python -c "from modelo_sentiment import IMDBSentimentClassifier; print('OK')"

# Verificar imports de la API
cd api && python -c "from api_sentiment import app; print('OK')"

# Ejecutar tests (si existen)
pytest tests/

# Ver estadísticas del modelo
cat results/statistics/model_statistics.json | jq '.'

# Contar líneas de código
find . -name "*.py" -not -path "./__pycache__/*" | xargs wc -l
```

## Contribuir

Para contribuir al proyecto:

1. Mantén la estructura de directorios
2. Documenta cambios en README.md
3. Actualiza requirements.txt si añades dependencias
4. Sigue el estilo de código existente
5. Añade tests para nuevas funcionalidades

## Soporte

Para preguntas o problemas:
- Abre un issue en GitHub
- Revisa la documentación en README.md
- Consulta README_API.md para la API
