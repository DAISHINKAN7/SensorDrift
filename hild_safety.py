import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

# ============================================
# CONFIGURATION
# ============================================
class Config:
    # Dataset parameters
    CSV_PATH = '/Users/kshitijnavale/Desktop/sensor data/synthetic_safety_dataset (1).csv'
    TEXT_COLUMN = 'text'
    LABEL_COLUMN = 'label'
    
    # Label weights (for weighted loss)
    LABEL_WEIGHTS = {
        'safe': 1000,
        'bullying': 700,
        'predatory': 600,
        'violence': 500,
        'self_harm': 400,
        'drugs': 300
    }
    
    # Model parameters
    MAX_WORDS = 10000  # Vocabulary size
    MAX_SEQUENCE_LENGTH = 100  # Max length of each text
    EMBEDDING_DIM = 128
    
    # Training parameters
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.15
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Output paths
    MODEL_PATH = 'safety_guardian_model.h5'
    TOKENIZER_PATH = 'tokenizer.pickle'
    LABEL_ENCODER_PATH = 'label_encoder.pickle'

# ============================================
# DATA PREPROCESSING
# ============================================
class DataPreprocessor:
    def __init__(self):
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?\'\"]', '', text)
        
        return text.strip()
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nLabel distribution:\n{df[Config.LABEL_COLUMN].value_counts()}")
        
        # Clean text
        print("\nCleaning text data...")
        df['cleaned_text'] = df[Config.TEXT_COLUMN].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        return df
    
    def encode_labels(self, labels):
        """Encode labels to numerical format"""
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels):
        """Decode numerical labels back to original"""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def prepare_text_data(self, texts, fit=True):
        """Tokenize and pad text sequences"""
        if fit:
            self.tokenizer = Tokenizer(num_words=Config.MAX_WORDS, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, 
                                        maxlen=Config.MAX_SEQUENCE_LENGTH,
                                        padding='post',
                                        truncating='post')
        return padded_sequences
    
    def calculate_class_weights(self, y_train):
        """Calculate class weights for imbalanced dataset"""
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        
        return dict(enumerate(class_weights))

# ============================================
# MODEL ARCHITECTURE
# ============================================
class SafetyGuardianModel:
    def __init__(self, num_classes, vocab_size):
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.model = None
        
    def build_lstm_model(self):
        """Build LSTM-based model for text classification"""
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=Config.EMBEDDING_DIM,
                input_length=Config.MAX_SEQUENCE_LENGTH
            ),
            
            # Spatial Dropout
            layers.SpatialDropout1D(0.3),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.4),
            
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.4),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_cnn_model(self):
        """Build CNN-based model for text classification"""
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=Config.EMBEDDING_DIM,
                input_length=Config.MAX_SEQUENCE_LENGTH
            ),
            
            # Convolutional layers
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_hybrid_model(self):
        """Build hybrid CNN-LSTM model"""
        model = keras.Sequential([
            # Embedding layer
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=Config.EMBEDDING_DIM,
                input_length=Config.MAX_SEQUENCE_LENGTH
            ),
            
            # CNN layers
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(pool_size=2),
            
            # LSTM layers
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=Config.LEARNING_RATE):
        """Compile the model"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE")
        print("="*60)
        self.model.summary()
        
    def train(self, X_train, y_train, X_val, y_val, class_weights=None):
        """Train the model"""
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

# ============================================
# EVALUATION AND VISUALIZATION
# ============================================
class ModelEvaluator:
    @staticmethod
    def plot_training_history(history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, label_encoder):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        # Classification Report
        print("\n" + "-"*60)
        print("CLASSIFICATION REPORT")
        print("-"*60)
        print(classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            digits=4
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, y_pred, y_pred_proba

# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main():
    print("="*60)
    print("AI-POWERED CHILD SAFETY GUARDIAN")
    print("Model Training Pipeline")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_and_preprocess_data(Config.CSV_PATH)
    
    # Prepare features and labels
    X = df['cleaned_text'].values
    y = preprocessor.encode_labels(df[Config.LABEL_COLUMN].values)
    
    num_classes = len(preprocessor.label_encoder.classes_)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {preprocessor.label_encoder.classes_}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE,
        random_state=42,
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=Config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_train
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    
    # Tokenize and pad sequences
    X_train_seq = preprocessor.prepare_text_data(X_train, fit=True)
    X_val_seq = preprocessor.prepare_text_data(X_val, fit=False)
    X_test_seq = preprocessor.prepare_text_data(X_test, fit=False)
    
    # Calculate class weights
    class_weights = preprocessor.calculate_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    # Build model
    vocab_size = min(Config.MAX_WORDS, len(preprocessor.tokenizer.word_index) + 1)
    
    safety_model = SafetyGuardianModel(num_classes, vocab_size)
    
    # Choose model architecture (default: hybrid)
    print("\nBuilding Hybrid CNN-LSTM model...")
    safety_model.build_hybrid_model()
    
    # Compile model
    safety_model.compile_model()
    
    # Train model
    history = safety_model.train(
        X_train_seq, y_train,
        X_val_seq, y_val,
        class_weights=class_weights
    )
    
    # Plot training history
    ModelEvaluator.plot_training_history(history)
    
    # Evaluate model
    accuracy, y_pred, y_pred_proba = ModelEvaluator.evaluate_model(
        safety_model.model,
        X_test_seq,
        y_test,
        preprocessor.label_encoder
    )
    
    # Save model and preprocessors
    print("\n" + "="*60)
    print("SAVING MODEL AND ARTIFACTS")
    print("="*60)
    
    safety_model.model.save(Config.MODEL_PATH)
    print(f"✓ Model saved to: {Config.MODEL_PATH}")
    
    with open(Config.TOKENIZER_PATH, 'wb') as f:
        pickle.dump(preprocessor.tokenizer, f)
    print(f"✓ Tokenizer saved to: {Config.TOKENIZER_PATH}")
    
    with open(Config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(preprocessor.label_encoder, f)
    print(f"✓ Label encoder saved to: {Config.LABEL_ENCODER_PATH}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()