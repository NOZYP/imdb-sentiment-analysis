from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pickle
from pathlib import Path
from typing import Dict
import json
import sys

# Add model directory to path
sys.path.append(str(Path(__file__).parent.parent / 'model'))
from modelo_sentiment import IMDBSentimentClassifier

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="API para análisis de sentimiento de reseñas de películas con ratings 1-10",
    version="1.0.0"
)

# Load model and initialize classifier
MODEL_PATH = "../results/models/imdb_sentiment_model.pkl"
DATA_PATH = r'c:\Users\crisc\OneDrive\Documentos\Codigos\Ciencia_de_datos\DatosNoEstructurados\Proyecto\aclImdb'

classifier = IMDBSentimentClassifier(DATA_PATH)
try:
    with open(MODEL_PATH, 'rb') as f:
        classifier.model = pickle.load(f)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠ Model file not found at {MODEL_PATH}. Please train the model first.")
    classifier = None

# Statistics path
STATS_PATH = "../results/statistics/model_statistics.json"
model_stats = None

class ReviewRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_label: int
    rating: int
    confidence: float
    probabilities: Dict[str, float]

@app.get("/", response_class=HTMLResponse)
async def home():
    """Página principal con interfaz interactiva"""
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IMDB Sentiment Analysis</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            header {
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }
            h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                font-size: 1.1em;
                opacity: 0.9;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            .tab-button {
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                transition: all 0.3s;
            }
            .tab-button:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            .tab-button.active {
                background: white;
                color: #667eea;
                font-weight: bold;
            }
            .tab-content {
                display: none;
                background: white;
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            .tab-content.active {
                display: block;
            }
            .section-title {
                font-size: 1.5em;
                color: #333;
                margin-bottom: 20px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 8px;
                color: #555;
                font-weight: 500;
            }
            textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
                font-family: inherit;
                resize: vertical;
                min-height: 120px;
            }
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            button[type="submit"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 32px;
                border: none;
                border-radius: 8px;
                font-size: 1em;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button[type="submit"]:hover {
                transform: translateY(-2px);
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                display: none;
            }
            .result.show {
                display: block;
            }
            .result-item {
                margin: 10px 0;
                padding: 10px;
                background: white;
                border-radius: 6px;
            }
            .sentiment-positive {
                color: #28a745;
                font-weight: bold;
            }
            .sentiment-negative {
                color: #dc3545;
                font-weight: bold;
            }
            .rating-stars {
                font-size: 1.5em;
                color: #ffc107;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }
            .stat-label {
                opacity: 0.9;
                font-size: 0.9em;
            }
            .distribution-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .distribution-table th,
            .distribution-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .distribution-table th {
                background: #667eea;
                color: white;
            }
            .distribution-table tr:hover {
                background: #f8f9fa;
            }
            .loading {
                text-align: center;
                padding: 20px;
                color: #667eea;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
            }
            .examples {
                margin-top: 20px;
            }
            .example-btn {
                background: #e9ecef;
                border: none;
                padding: 8px 16px;
                margin: 5px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.9em;
            }
            .example-btn:hover {
                background: #dee2e6;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎬 IMDB Sentiment Analysis</h1>
                <p class="subtitle">Análisis de sentimiento de reseñas con rating 1-10 estrellas</p>
            </header>

            <div class="tabs">
                <button class="tab-button active" onclick="showTab('predict')">Predicción</button>
                <button class="tab-button" onclick="showTab('stats')">Estadísticas</button>
                <button class="tab-button" onclick="showTab('distribution')">Distribución</button>
            </div>

            <!-- Tab: Predicción -->
            <div id="predict-tab" class="tab-content active">
                <h2 class="section-title">Analizar Reseña</h2>
                <form id="prediction-form">
                    <div class="form-group">
                        <label for="review-text">Escribe una reseña de película:</label>
                        <textarea id="review-text" name="text" placeholder="Ej: This movie was absolutely fantastic! The acting was brilliant..."></textarea>
                    </div>
                    <button type="submit">Analizar Sentimiento</button>
                </form>

                <div class="examples">
                    <strong>Ejemplos rápidos:</strong><br>
                    <button class="example-btn" onclick="setExample('positive')">Reseña Positiva</button>
                    <button class="example-btn" onclick="setExample('negative')">Reseña Negativa</button>
                    <button class="example-btn" onclick="setExample('neutral')">Reseña Neutral</button>
                </div>

                <div id="prediction-result" class="result">
                    <h3>Resultado del Análisis</h3>
                    <div class="result-item">
                        <strong>Sentimiento:</strong> <span id="sentiment"></span>
                    </div>
                    <div class="result-item">
                        <strong>Rating:</strong> <span id="rating" class="rating-stars"></span>
                    </div>
                    <div class="result-item">
                        <strong>Confianza:</strong> <span id="confidence"></span>
                    </div>
                    <div class="result-item">
                        <strong>Probabilidades:</strong><br>
                        Negativo: <span id="prob-neg"></span><br>
                        Positivo: <span id="prob-pos"></span>
                    </div>
                </div>
            </div>

            <!-- Tab: Estadísticas -->
            <div id="stats-tab" class="tab-content">
                <h2 class="section-title">Estadísticas del Modelo</h2>
                <div id="stats-loading" class="loading">Cargando estadísticas...</div>
                <div id="stats-content" style="display: none;">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-label">Accuracy</div>
                            <div class="stat-value" id="stat-accuracy">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">F1 Score</div>
                            <div class="stat-value" id="stat-f1">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Precisión</div>
                            <div class="stat-value" id="stat-precision">-</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Recall</div>
                            <div class="stat-value" id="stat-recall">-</div>
                        </div>
                    </div>
                    <h3>Matriz de Confusión</h3>
                    <table class="distribution-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>Predicho Negativo</th>
                                <th>Predicho Positivo</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Real Negativo</strong></td>
                                <td id="cm-tn">-</td>
                                <td id="cm-fp">-</td>
                            </tr>
                            <tr>
                                <td><strong>Real Positivo</strong></td>
                                <td id="cm-fn">-</td>
                                <td id="cm-tp">-</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Tab: Distribución -->
            <div id="distribution-tab" class="tab-content">
                <h2 class="section-title">Distribución de Datos</h2>
                <div id="dist-loading" class="loading">Cargando distribución...</div>
                <div id="dist-content" style="display: none;">
                    <h3>Distribución de Ratings - Conjunto de Entrenamiento</h3>
                    <table class="distribution-table">
                        <thead>
                            <tr>
                                <th>Sentimiento</th>
                                <th>Cantidad de Reseñas</th>
                                <th>Rating Promedio</th>
                                <th>Desviación Estándar</th>
                                <th>Rango</th>
                            </tr>
                        </thead>
                        <tbody id="dist-table-body-train">
                        </tbody>
                    </table>
                    
                    <h3 style="margin-top: 30px;">Distribución de Ratings - Conjunto de Test</h3>
                    <table class="distribution-table">
                        <thead>
                            <tr>
                                <th>Sentimiento</th>
                                <th>Cantidad de Reseñas</th>
                                <th>Rating Promedio</th>
                                <th>Desviación Estándar</th>
                                <th>Rango</th>
                            </tr>
                        </thead>
                        <tbody id="dist-table-body-test">
                        </tbody>
                    </table>
                    
                    <h3 style="margin-top: 30px;">Accuracy por Rating (Conjunto de Test)</h3>
                    <table class="distribution-table">
                        <thead>
                            <tr>
                                <th>Rating</th>
                                <th>Accuracy</th>
                                <th>Cantidad</th>
                            </tr>
                        </thead>
                        <tbody id="rating-accuracy-body">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            const examples = {
                positive: "This movie was absolutely fantastic! The best film I've seen in years. The acting was superb, the story was engaging, and the cinematography was stunning. I highly recommend it to everyone!",
                negative: "Worst movie ever made. Horrible acting, terrible plot, and boring from start to finish. Complete waste of time and money. I want my money back!",
                neutral: "It was okay, not great but not terrible either. Some parts were good but overall just average."
            };

            function showTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });

                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');

                // Load data if needed
                if (tabName === 'stats') {
                    loadStats();
                } else if (tabName === 'distribution') {
                    loadDistribution();
                }
            }

            function setExample(type) {
                document.getElementById('review-text').value = examples[type];
            }

            document.getElementById('prediction-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                const text = document.getElementById('review-text').value;
                
                if (!text.trim()) {
                    alert('Por favor escribe una reseña');
                    return;
                }

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: text })
                    });

                    if (!response.ok) {
                        throw new Error('Error en la predicción');
                    }

                    const data = await response.json();
                    
                    // Show results
                    const resultDiv = document.getElementById('prediction-result');
                    resultDiv.classList.add('show');
                    
                    const sentimentClass = data.sentiment === 'Positive' ? 'sentiment-positive' : 'sentiment-negative';
                    document.getElementById('sentiment').innerHTML = `<span class="${sentimentClass}">${data.sentiment}</span>`;
                    
                    const stars = '⭐'.repeat(data.rating);
                    document.getElementById('rating').textContent = `${stars} (${data.rating}/10)`;
                    
                    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(2)}%`;
                    document.getElementById('prob-neg').textContent = `${(data.probabilities.negative * 100).toFixed(2)}%`;
                    document.getElementById('prob-pos').textContent = `${(data.probabilities.positive * 100).toFixed(2)}%`;
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            });

            async function loadStats() {
                const loading = document.getElementById('stats-loading');
                const content = document.getElementById('stats-content');
                
                try {
                    const response = await fetch('/statistics');
                    if (!response.ok) throw new Error('Error loading statistics');
                    
                    const data = await response.json();
                    
                    document.getElementById('stat-accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
                    document.getElementById('stat-f1').textContent = data.f1_score.toFixed(4);
                    document.getElementById('stat-precision').textContent = data.precision.toFixed(4);
                    document.getElementById('stat-recall').textContent = data.recall.toFixed(4);
                    
                    // Confusion matrix
                    const cm = data.confusion_matrix;
                    document.getElementById('cm-tn').textContent = cm[0][0];
                    document.getElementById('cm-fp').textContent = cm[0][1];
                    document.getElementById('cm-fn').textContent = cm[1][0];
                    document.getElementById('cm-tp').textContent = cm[1][1];
                    
                    loading.style.display = 'none';
                    content.style.display = 'block';
                } catch (error) {
                    loading.innerHTML = `<div class="error">Error cargando estadísticas: ${error.message}</div>`;
                }
            }

            async function loadDistribution() {
                const loading = document.getElementById('dist-loading');
                const content = document.getElementById('dist-content');
                
                try {
                    const response = await fetch('/distribution');
                    if (!response.ok) throw new Error('Error loading distribution');
                    
                    const data = await response.json();
                    
                    // Training distribution
                    const distTableBodyTrain = document.getElementById('dist-table-body-train');
                    distTableBodyTrain.innerHTML = '';
                    
                    for (const [sentiment, info] of Object.entries(data.training_distribution)) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td><strong>${sentiment}</strong></td>
                            <td>${info.count}</td>
                            <td>${info.mean_rating.toFixed(2)}</td>
                            <td>${info.std_rating.toFixed(2)}</td>
                            <td>${info.min_rating} - ${info.max_rating}</td>
                        `;
                        distTableBodyTrain.appendChild(row);
                    }
                    
                    // Test distribution
                    const distTableBodyTest = document.getElementById('dist-table-body-test');
                    distTableBodyTest.innerHTML = '';
                    
                    if (data.test_distribution) {
                        for (const [sentiment, info] of Object.entries(data.test_distribution)) {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td><strong>${sentiment}</strong></td>
                                <td>${info.count}</td>
                                <td>${info.mean_rating.toFixed(2)}</td>
                                <td>${info.std_rating.toFixed(2)}</td>
                                <td>${info.min_rating} - ${info.max_rating}</td>
                            `;
                            distTableBodyTest.appendChild(row);
                        }
                    }
                    
                    // Rating accuracy
                    const ratingTableBody = document.getElementById('rating-accuracy-body');
                    ratingTableBody.innerHTML = '';
                    
                    for (const [rating, info] of Object.entries(data.rating_accuracy)) {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td><strong>Rating ${rating}</strong></td>
                            <td>${(info.accuracy * 100).toFixed(2)}%</td>
                            <td>${info.count}</td>
                        `;
                        ratingTableBody.appendChild(row);
                    }
                    
                    loading.style.display = 'none';
                    content.style.display = 'block';
                } catch (error) {
                    loading.innerHTML = `<div class="error">Error cargando distribución: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(review: ReviewRequest):
    """Predice el sentimiento y rating de una reseña"""
    if classifier is None or classifier.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        sentiment, rating, proba = classifier.predict_sentiment_with_rating(review.text)
        
        return PredictionResponse(
            text=review.text,
            sentiment="Positive" if sentiment == 1 else "Negative",
            sentiment_label=int(sentiment),
            rating=int(rating),
            confidence=float(max(proba)),
            probabilities={
                "negative": float(proba[0]),
                "positive": float(proba[1])
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/statistics")
async def get_statistics():
    """Obtiene las estadísticas del modelo"""
    global model_stats
    
    # Try to load from file first
    if model_stats is None:
        try:
            with open(STATS_PATH, 'r') as f:
                model_stats = json.load(f)
                print(f"✓ Statistics loaded from {STATS_PATH}")
        except FileNotFoundError:
            raise HTTPException(
                status_code=404, 
                detail="Statistics not found. Please train and evaluate the model first by running modelo_sentiment.py"
            )
    
    return JSONResponse(content=model_stats)

@app.get("/distribution")
async def get_distribution():
    """Obtiene la distribución de datos de entrenamiento y test"""
    global model_stats
    
    if model_stats is None:
        try:
            with open(STATS_PATH, 'r') as f:
                model_stats = json.load(f)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Statistics not found. Please train and evaluate the model first."
            )
    
    return JSONResponse(content={
        "training_distribution": model_stats.get("training_distribution", {}),
        "test_distribution": model_stats.get("test_distribution", {}),
        "rating_accuracy": model_stats.get("rating_accuracy", {})
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats_exist = Path(STATS_PATH).exists()
    return {
        "status": "healthy",
        "model_loaded": classifier is not None and classifier.model is not None,
        "statistics_available": stats_exist
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("Starting IMDB Sentiment Analysis API")
    print("="*50)
    print("\nAccess the application at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
