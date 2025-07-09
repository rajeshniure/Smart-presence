import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import glob
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """
    Emotion Classification model using a manually coded CNN
    """
    
    def __init__(self, model_path=None, input_size=(48, 48)):
        self.input_size = input_size
        self.model = None
        self.trained = False
        self.emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.num_classes = len(self.emotion_labels)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self):
        """
        Create a CNN model for emotion classification
        Manually coded architecture optimized for emotion recognition
        """
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(*self.input_size, 3)),
            
            # First Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling instead of Flatten for better generalization
            layers.GlobalAveragePooling2D(),
            
            # Fully Connected Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='emotion_output')
        ])
        
        return model
    
    def load_data(self, data_dir, validation_split=0.2):
        """
        Load and preprocess the emotion dataset
        """
        logger.info("Loading emotion dataset...")
        
        images = []
        labels = []
        
        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(data_dir, emotion)
            
            if not os.path.exists(emotion_dir):
                logger.warning(f"Emotion directory not found: {emotion_dir}")
                continue
                
            image_files = glob.glob(os.path.join(emotion_dir, "*.png"))
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale and then back to RGB for consistency
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
                    
                    # Resize to input size
                    img_resized = cv2.resize(img_rgb, self.input_size)
                    
                    # Normalize to [0, 1]
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    images.append(img_normalized)
                    labels.append(emotion_idx)
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            logger.info(f"Loaded {len([l for l in labels if l == emotion_idx])} samples for {emotion}")
        
        if len(images) == 0:
            raise ValueError("No valid images found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Convert labels to categorical
        y_categorical = keras.utils.to_categorical(y, self.num_classes)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y_categorical
        )
        
        logger.info(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")
        logger.info(f"Data shape: {X.shape}, Labels shape: {y_categorical.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def create_data_augmentation(self):
        """
        Create data augmentation pipeline
        """
        data_augmentation = keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        
        return data_augmentation
    
    def train(self, data_dir, epochs=100, batch_size=32, learning_rate=0.001, use_augmentation=True):
        """
        Train the emotion classification model
        """
        logger.info("Starting emotion classification model training...")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data(data_dir)
        
        # Create model
        self.model = self.create_model()
        
        # Add data augmentation if requested
        if use_augmentation:
            data_augmentation = self.create_data_augmentation()
            # Insert augmentation at the beginning of the model
            augmented_model = keras.Sequential([
                data_augmentation,
                self.model
            ])
            self.model = augmented_model
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True, 
                monitor='val_accuracy'
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=7, 
                min_lr=1e-7,
                monitor='val_loss'
            ),
            keras.callbacks.ModelCheckpoint(
                'models/emotion_detection_best.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            )
        ]
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.trained = True
        
        # Plot training history
        self._plot_training_history(history)
        
        # Generate classification report
        self._generate_classification_report(X_val, y_val)
        
        return history
    
    def _plot_training_history(self, history):
        """
        Plot training history
        """
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot top-3 accuracy
        plt.subplot(1, 3, 3)
        plt.plot(history.history['top_3_accuracy'], label='Training Top-3 Accuracy')
        plt.plot(history.history['val_top_3_accuracy'], label='Validation Top-3 Accuracy')
        plt.title('Model Top-3 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Top-3 Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/emotion_detection_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_classification_report(self, X_val, y_val):
        """
        Generate and plot classification report and confusion matrix
        """
        # Get predictions
        y_pred = self.model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Classification report
        report = classification_report(
            y_true_classes, 
            y_pred_classes, 
            target_names=self.emotion_labels,
            output_dict=True
        )
        
        logger.info("Classification Report:")
        logger.info(classification_report(y_true_classes, y_pred_classes, target_names=self.emotion_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.emotion_labels,
            yticklabels=self.emotion_labels
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('models/emotion_detection_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save classification report
        import json
        with open('models/emotion_classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def predict(self, image):
        """
        Predict emotion in an image
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
        
        if image is None:
            raise ValueError("Could not load image")
        
        # Convert to grayscale and then to RGB
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image
            
        image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        
        # Resize and normalize
        img_resized = cv2.resize(image_rgb, self.input_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        
        # Get top prediction
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])
        emotion = self.emotion_labels[predicted_class]
        
        # Get all predictions for detailed analysis
        emotion_scores = {
            self.emotion_labels[i]: float(predictions[i]) 
            for i in range(len(self.emotion_labels))
        }
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_scores': emotion_scores
        }
    
    def predict_batch(self, images):
        """
        Predict emotions for a batch of images
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        processed_images = []
        
        for image in images:
            if isinstance(image, str):
                image = cv2.imread(image)
            
            if image is None:
                continue
                
            # Convert to grayscale and then to RGB
            if len(image.shape) == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                image_gray = image
                
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
            
            # Resize and normalize
            img_resized = cv2.resize(image_rgb, self.input_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            processed_images.append(img_normalized)
        
        if not processed_images:
            return []
        
        # Predict batch
        predictions = self.model.predict(np.array(processed_images), verbose=0)
        
        results = []
        for pred in predictions:
            predicted_class = np.argmax(pred)
            confidence = float(pred[predicted_class])
            emotion = self.emotion_labels[predicted_class]
            
            emotion_scores = {
                self.emotion_labels[i]: float(pred[i]) 
                for i in range(len(self.emotion_labels))
            }
            
            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'all_scores': emotion_scores
            })
        
        return results
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        """
        self.model = keras.models.load_model(filepath)
        self.trained = True
        logger.info(f"Model loaded from {filepath}")

def train_emotion_detection_model():
    """
    Main function to train the emotion detection model
    """
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Path to dataset
    train_data_dir = "datasets/Emotion/train"
    
    # Train model
    history = classifier.train(
        data_dir=train_data_dir,
        epochs=150,
        batch_size=64,
        learning_rate=0.001,
        use_augmentation=True
    )
    
    # Save model
    classifier.save_model("models/emotion_detection_model.h5")
    
    # Test on test set if available
    test_data_dir = "datasets/Emotion/test"
    if os.path.exists(test_data_dir):
        logger.info("Evaluating on test set...")
        X_test, _, y_test, _ = classifier.load_data(test_data_dir, validation_split=0.0)
        
        if len(X_test) > 0:
            test_loss, test_accuracy, test_top3_accuracy = classifier.model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Test Loss: {test_loss:.4f}")
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")
            logger.info(f"Test Top-3 Accuracy: {test_top3_accuracy:.4f}")
    
    return classifier

if __name__ == "__main__":
    train_emotion_detection_model() 