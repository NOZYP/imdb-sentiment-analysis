import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score
)
from sklearn.pipeline import Pipeline

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class IMDBSentimentClassifier:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.train_path = self.data_path / 'train'
        self.test_path = self.data_path / 'test'
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Model pipeline
        self.model = None
        
        # Statistics storage
        self.statistics = None
        self.train_data_info = None
        self.test_data_info = None
        
    def load_reviews_from_folder(self, folder_path, label=None):
        """Load reviews from a folder"""
        reviews = []
        labels = []
        ratings = []
        
        folder = Path(folder_path)
        files = list(folder.glob('*.txt'))
        
        print(f"Loading {len(files)} reviews from {folder_path}...")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    reviews.append(text)
                    
                    if label is not None:
                        labels.append(label)
                        
                        # Extract rating from filename (format: "1234_7.txt" means rating 7)
                        filename = file_path.stem
                        rating = None
                        if '_' in filename:
                            try:
                                rating = int(filename.split('_')[1])
                            except:
                                pass
                        
                        if rating is None:
                            # Fallback: estimate based on label
                            rating = 8 if label == 1 else 3
                        
                        ratings.append(rating)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return reviews, labels, ratings
    
    def preprocess_text(self, text):
        """Preprocess text using NLTK"""
        # Use NLTK for preprocessing
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def load_dataset(self):
        """Load training and test datasets"""
        print("\n=== Loading Dataset ===")
        
        # Load training data
        train_pos_reviews, train_pos_labels, train_pos_ratings = self.load_reviews_from_folder(
            self.train_path / 'pos', label=1
        )
        train_neg_reviews, train_neg_labels, train_neg_ratings = self.load_reviews_from_folder(
            self.train_path / 'neg', label=0
        )
        
        # Load test data
        test_pos_reviews, test_pos_labels, test_pos_ratings = self.load_reviews_from_folder(
            self.test_path / 'pos', label=1
        )
        test_neg_reviews, test_neg_labels, test_neg_ratings = self.load_reviews_from_folder(
            self.test_path / 'neg', label=0
        )
        
        # Combine data
        X_train = train_pos_reviews + train_neg_reviews
        y_train = train_pos_labels + train_neg_labels
        ratings_train = train_pos_ratings + train_neg_ratings
        
        X_test = test_pos_reviews + test_neg_reviews
        y_test = test_pos_labels + test_neg_labels
        ratings_test = test_pos_ratings + test_neg_ratings
        
        print(f"\nTraining samples: {len(X_train)} (Positive: {sum(y_train)}, Negative: {len(y_train) - sum(y_train)})")
        print(f"Test samples: {len(X_test)} (Positive: {sum(y_test)}, Negative: {len(y_test) - sum(y_test)})")
        
        # Store dataset distribution information
        self._store_dataset_info(y_train, ratings_train, y_test, ratings_test)
        
        return X_train, y_train, ratings_train, X_test, y_test, ratings_test
    
    def _store_dataset_info(self, y_train, ratings_train, y_test, ratings_test):
        """Store training and test dataset distribution information"""
        ratings_array_train = np.array(ratings_train)
        y_train_array = np.array(y_train)
        
        pos_ratings_train = ratings_array_train[y_train_array == 1]
        neg_ratings_train = ratings_array_train[y_train_array == 0]
        
        self.train_data_info = {
            "Positive": {
                "count": int(sum(y_train)),
                "mean_rating": float(pos_ratings_train.mean()),
                "std_rating": float(pos_ratings_train.std()),
                "min_rating": int(pos_ratings_train.min()),
                "max_rating": int(pos_ratings_train.max())
            },
            "Negative": {
                "count": int(len(y_train) - sum(y_train)),
                "mean_rating": float(neg_ratings_train.mean()),
                "std_rating": float(neg_ratings_train.std()),
                "min_rating": int(neg_ratings_train.min()),
                "max_rating": int(neg_ratings_train.max())
            }
        }
        
        ratings_array_test = np.array(ratings_test)
        y_test_array = np.array(y_test)
        
        pos_ratings_test = ratings_array_test[y_test_array == 1]
        neg_ratings_test = ratings_array_test[y_test_array == 0]
        
        self.test_data_info = {
            "Positive": {
                "count": int(sum(y_test)),
                "mean_rating": float(pos_ratings_test.mean()),
                "std_rating": float(pos_ratings_test.std()),
                "min_rating": int(pos_ratings_test.min()),
                "max_rating": int(pos_ratings_test.max())
            },
            "Negative": {
                "count": int(len(y_test) - sum(y_test)),
                "mean_rating": float(neg_ratings_test.mean()),
                "std_rating": float(neg_ratings_test.std()),
                "min_rating": int(neg_ratings_test.min()),
                "max_rating": int(neg_ratings_test.max())
            }
        }
    
    def preprocess_dataset(self, X_train, X_test):
        """Preprocess all reviews in the dataset"""
        print(f"\n=== Preprocessing Text (NLTK) ===")
        
        X_train_processed = []
        for i, text in enumerate(X_train):
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(X_train)} training reviews...")
            X_train_processed.append(self.preprocess_text(text))
        
        X_test_processed = []
        for i, text in enumerate(X_test):
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(X_test)} test reviews...")
            X_test_processed.append(self.preprocess_text(text))
        
        print("Preprocessing complete!")
        return X_train_processed, X_test_processed
    
    def train_model(self, X_train, y_train, ratings_train=None):
        """Train the sentiment classification model with optional rating-aware optimization"""
        print("\n=== Training Model ===")
        
        # Create pipeline with TF-IDF vectorizer and Logistic Regression
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=5, max_df=0.8)),
            ('classifier', LogisticRegression(max_iter=1000, C=0.5, random_state=42, class_weight='balanced'))
        ])
        
        # Train the model
        print("Training Logistic Regression model...")
        self.model.fit(X_train, y_train)
        
        # If ratings are provided, show distribution analysis
        if ratings_train is not None:
            print("\n=== Rating Distribution in Training Set ===")
            ratings_array = np.array(ratings_train)
            y_train_array = np.array(y_train)
            
            print(f"Positive reviews (label=1):")
            pos_ratings = ratings_array[y_train_array == 1]
            print(f"  Mean rating: {pos_ratings.mean():.2f}, Std: {pos_ratings.std():.2f}")
            print(f"  Range: {pos_ratings.min()}-{pos_ratings.max()}")
            
            print(f"Negative reviews (label=0):")
            neg_ratings = ratings_array[y_train_array == 0]
            print(f"  Mean rating: {neg_ratings.mean():.2f}, Std: {neg_ratings.std():.2f}")
            print(f"  Range: {neg_ratings.min()}-{neg_ratings.max()}")
        
        # Cross-validation
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("Training complete!")
    
    def evaluate_model(self, X_test, y_test, ratings_test=None):
        """Evaluate the model on test data with optional rating analysis"""
        print("\n=== Evaluating Model ===")
        
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
        
        print("\nConfusion Matrix:")
        print(cm)
        
        # Initialize statistics dictionary
        rating_accuracy_dict = {}
        
        # Rating-based analysis if ratings are available
        if ratings_test is not None:
            print("\n=== Rating-Based Analysis ===")
            y_test_array = np.array(y_test)
            ratings_array = np.array(ratings_test)
            y_pred_array = np.array(y_pred)
            
            # Analyze predictions per rating group
            for rating in range(1, 11):
                mask = ratings_array == rating
                if mask.sum() > 0:
                    rating_accuracy = accuracy_score(y_test_array[mask], y_pred_array[mask])
                    count = mask.sum()
                    rating_accuracy_dict[str(rating)] = {
                        "accuracy": float(rating_accuracy),
                        "count": int(count)
                    }
                    print(f"Rating {rating:2d}: Accuracy {rating_accuracy:.4f} ({count} reviews)")
            
            # Check if low ratings get negative predictions and high ratings get positive
            print("\n=== Sentiment Alignment Check ===")
            low_rating_mask = ratings_array <= 4
            high_rating_mask = ratings_array >= 7
            
            if low_rating_mask.sum() > 0:
                low_rating_negative_pct = (y_pred_array[low_rating_mask] == 0).sum() / low_rating_mask.sum() * 100
                print(f"Low ratings (1-4): {low_rating_negative_pct:.1f}% predicted as Negative")
            
            if high_rating_mask.sum() > 0:
                high_rating_positive_pct = (y_pred_array[high_rating_mask] == 1).sum() / high_rating_mask.sum() * 100
                print(f"High ratings (7-10): {high_rating_positive_pct:.1f}% predicted as Positive")
        
        # Store statistics
        self.statistics = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cm.tolist(),
            "rating_accuracy": rating_accuracy_dict,
            "training_distribution": self.train_data_info if self.train_data_info else {},
            "test_distribution": self.test_data_info if self.test_data_info else {}
        }
        
        return accuracy, f1
    
    def predict_sentiment_with_rating(self, text):
        """Predict sentiment and generate 1-10 star rating based on sentiment intensity"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get prediction probability
        proba = self.model.predict_proba([processed_text])[0]
        sentiment = self.model.predict([processed_text])[0]
        
        # Convert probability to 1-10 rating scale based on sentiment INTENSITY
        # Use probability as measure of how positive/negative the text is
        # proba[1] = probability of being positive (ranges 0 to 1)
        
        # Map the full probability spectrum to 1-10 scale:
        # proba[1] close to 0 = very negative = rating 1
        # proba[1] = 0.5 = neutral boundary = rating 5-6
        # proba[1] close to 1 = very positive = rating 10
        
        # More nuanced mapping based on observed data distributions:
        # Negative reviews (actual): mean=2.22, range 1-4
        # Positive reviews (actual): mean=8.74, range 7-10
        
        pos_prob = proba[1]
        
        if pos_prob < 0.5:
            # Negative sentiment: map 0.0-0.5 to ratings 1-4
            # More negative (lower prob) -> lower rating
            normalized = pos_prob / 0.5  # Scale 0.0-0.5 to 0.0-1.0
            rating = int(1 + normalized * 3)  # Maps to 1-4
            rating = max(1, min(4, rating))
        else:
            # Positive sentiment: map 0.5-1.0 to ratings 7-10
            # More positive (higher prob) -> higher rating
            normalized = (pos_prob - 0.5) / 0.5  # Scale 0.5-1.0 to 0.0-1.0
            rating = int(7 + normalized * 3)  # Maps to 7-10
            rating = max(7, min(10, rating))
        
        return sentiment, rating, proba
    
    def process_unlabeled_data(self):
        """Process unlabeled reviews from the unsup folder"""
        print("\n=== Processing Unlabeled Data ===")
        
        if self.model is None:
            print("Error: Model not trained. Please train the model first.")
            return None
        
        unsup_path = self.train_path / 'unsup'
        if not unsup_path.exists():
            print("No unlabeled data folder found!")
            return None
        
        # Load unlabeled reviews
        unsup_reviews, _, _ = self.load_reviews_from_folder(unsup_path)
        
        print(f"Processing {len(unsup_reviews)} unlabeled reviews...")
        
        results = []
        for i, review in enumerate(unsup_reviews):
            if (i + 1) % 5000 == 0:
                print(f"Processed {i + 1}/{len(unsup_reviews)} unlabeled reviews...")
            
            sentiment, rating, proba = self.predict_sentiment_with_rating(review)
            results.append({
                'review': review[:200] + '...' if len(review) > 200 else review,
                'sentiment': 'Positive' if sentiment == 1 else 'Negative',
                'rating': rating,
                'confidence': max(proba)
            })
        
        # Create DataFrame
        df_results = pd.DataFrame(results)
        
        print("\n=== Unlabeled Data Processing Results ===")
        print(f"Total reviews processed: {len(df_results)}")
        print(f"Positive reviews: {sum(df_results['sentiment'] == 'Positive')}")
        print(f"Negative reviews: {sum(df_results['sentiment'] == 'Negative')}")
        print(f"\nRating distribution:")
        print(df_results['rating'].value_counts().sort_index())
        print(f"\nAverage confidence: {df_results['confidence'].mean():.4f}")
        
        # Save results
        output_file = '../results/predictions/unlabeled_predictions.csv'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return df_results
    
    def save_model(self, filename='../results/models/imdb_sentiment_model.pkl'):
        """Save the trained model"""
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to {filename}")
    
    def save_statistics(self, filename='../results/statistics/model_statistics.json'):
        """Save model statistics to JSON file"""
        if self.statistics is None:
            print("No statistics available. Please evaluate the model first.")
            return
        
        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.statistics, f, indent=2)
        print(f"Statistics saved to {filename}")
    
    def load_model(self, filename='../results/models/imdb_sentiment_model.pkl'):
        """Load a trained model"""
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filename}")
    
    def load_statistics(self, filename='../results/statistics/model_statistics.json'):
        """Load model statistics from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.statistics = json.load(f)
            print(f"Statistics loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Statistics file not found: {filename}")
            return False

def main():
    # Set the path to the IMDB dataset
    data_path = r'c:\Users\crisc\OneDrive\Documentos\Codigos\Ciencia_de_datos\DatosNoEstructurados\Proyecto\aclImdb'
    
    # Initialize classifier
    classifier = IMDBSentimentClassifier(data_path)
    
    # Load dataset
    X_train, y_train, ratings_train, X_test, y_test, ratings_test = classifier.load_dataset()
    
    # Preprocess dataset
    X_train_processed, X_test_processed = classifier.preprocess_dataset(X_train, X_test)
    
    # Train model with rating information
    classifier.train_model(X_train_processed, y_train, ratings_train)
    
    # Evaluate model with rating analysis
    accuracy, f1 = classifier.evaluate_model(X_test_processed, y_test, ratings_test)
    
    # Save statistics immediately after evaluation
    classifier.save_statistics()
    
    # Check if accuracy meets requirements (80-90%)
    if accuracy >= 0.80:
        print(f"\n✓ Model meets accuracy requirements! ({accuracy:.2%})")
        
        # Save the model
        classifier.save_model()
        
        # Test with sample predictions
        print("\n=== Sample Predictions ===")
        sample_reviews = [
            "This movie was absolutely fantastic! The best film I've seen in years.",
            "Terrible movie, waste of time. I want my money back.",
            "It was okay, not great but not terrible either.",
            "Masterpiece! Amazing acting, brilliant story, perfect execution!",
            "Worst movie ever made. Horrible in every way."
        ]
        
        for review in sample_reviews:
            sentiment, rating, proba = classifier.predict_sentiment_with_rating(review)
            print(f"\nReview: {review[:60]}...")
            print(f"Sentiment: {'Positive' if sentiment == 1 else 'Negative'}")
            print(f"Rating: {rating}/10 stars")
            print(f"Confidence: {max(proba):.4f}")
        
        # Process unlabeled data
        user_input = input("\n\nDo you want to process the unlabeled data now? (y/n): ")
        if user_input.lower() == 'y':
            classifier.process_unlabeled_data()
    else:
        print(f"\n✗ Model accuracy ({accuracy:.2%}) is below the required 80% threshold.")
        print("Consider tuning hyperparameters or trying different preprocessing methods.")
        # Still save statistics even if below threshold
        classifier.save_statistics()

if __name__ == "__main__":
    main()
