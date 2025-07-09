import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import json
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face Detection model using a CNN architecture similar to MTCNN
    """
    
    def __init__(self, model_path=None, input_size=(224, 224)):
        self.input_size = input_size
        self.model = None
        self.trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def create_model(self):
        """
        Create a CNN model for face detection
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.input_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Fifth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            
            # Output layers for bounding box regression and classification
            layers.Dense(5, activation='linear', name='bbox_output')  # 4 for bbox + 1 for confidence
        ])
        
        return model
    
    def load_data(self, images_dir, labels_dir, validation_split=0.2):
        """
        Load and preprocess the face detection dataset
        """
        logger.info("Loading face detection dataset...")
        
        images = []
        bboxes = []
        confidences = []
        
        # Get all image files
        image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
        
        for img_path in image_files:
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_size)
                img = img.astype(np.float32) / 255.0
                
                # Load corresponding label
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(labels_dir, f"{base_name}.txt")
                
                if not os.path.exists(label_path):
                    continue
                
                # Parse YOLO format labels
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines:
                    # Take the first face detection (you can modify to handle multiple faces)
                    line = lines[0].strip().split()
                    if len(line) >= 5:
                        class_id, center_x, center_y, width, height = map(float, line)
                        
                        # Convert to absolute coordinates
                        img_h, img_w = self.input_size
                        
                        # YOLO format to bbox format (x1, y1, x2, y2)
                        x1 = (center_x - width/2) * img_w
                        y1 = (center_y - height/2) * img_h
                        x2 = (center_x + width/2) * img_w
                        y2 = (center_y + height/2) * img_h
                        
                        # Normalize to [0, 1]
                        bbox = [x1/img_w, y1/img_h, x2/img_w, y2/img_h]
                        
                        images.append(img)
                        bboxes.append(bbox)
                        confidences.append(1.0)  # Face present
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        if len(images) == 0:
            raise ValueError("No valid images found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(images)
        y_bbox = np.array(bboxes)
        y_conf = np.array(confidences)
        
        # Combine bbox and confidence
        y = np.column_stack([y_bbox, y_conf])
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        logger.info(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples")
        
        return X_train, X_val, y_train, y_val
    
    def train(self, images_dir, labels_dir, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train the face detection model
        """
        logger.info("Starting face detection model training...")
        
        # Load data
        X_train, X_val, y_train, y_val = self.load_data(images_dir, labels_dir)
        
        # Create model
        self.model = self.create_model()
        
        # Compile model with custom loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=self._combined_loss,
            metrics=['mae']
        )
        
        # Print model summary
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            keras.callbacks.ModelCheckpoint(
                'models/face_detection_best.h5',
                save_best_only=True,
                monitor='val_loss'
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
        
        return history
    
    def _combined_loss(self, y_true, y_pred):
        """
        Combined loss for bounding box regression and face confidence
        """
        # Split predictions
        bbox_true = y_true[:, :4]  # x1, y1, x2, y2
        conf_true = y_true[:, 4:5]  # confidence
        
        bbox_pred = y_pred[:, :4]
        conf_pred = y_pred[:, 4:5]
        
        # Bounding box loss (Smooth L1 loss)
        bbox_loss = tf.keras.losses.Huber()(bbox_true, bbox_pred)
        
        # Confidence loss (Binary crossentropy)
        conf_loss = tf.keras.losses.BinaryCrossentropy()(conf_true, tf.nn.sigmoid(conf_pred))
        
        # Combine losses
        total_loss = bbox_loss + conf_loss
        
        return total_loss
    
    def _plot_training_history(self, history):
        """
        Plot training history
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('models/face_detection_training_history.png')
        plt.show()
    
    def predict(self, image):
        """
        Predict face location in an image
        """
        if not self.trained or self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_h, original_w = image.shape[:2]
        
        # Resize and normalize
        img_resized = cv2.resize(image, self.input_size)
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        prediction = self.model.predict(img_batch, verbose=0)[0]
        
        # Extract bbox and confidence
        bbox = prediction[:4]
        confidence = tf.nn.sigmoid(prediction[4]).numpy()
        
        # Convert normalized coordinates back to original image size
        x1 = int(bbox[0] * original_w)
        y1 = int(bbox[1] * original_h)
        x2 = int(bbox[2] * original_w)
        y2 = int(bbox[3] * original_h)
        
        return {
            'bbox': [x1, y1, x2, y2],
            'confidence': float(confidence)
        }
    
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
        self.model = keras.models.load_model(filepath, custom_objects={'_combined_loss': self._combined_loss})
        self.trained = True
        logger.info(f"Model loaded from {filepath}")

def train_face_detection_model():
    """
    Main function to train the face detection model
    """
    # Initialize detector
    detector = FaceDetector()
    
    # Paths to dataset
    train_images_dir = "datasets/face detection/images/train"
    train_labels_dir = "datasets/face detection/labels/train"
    
    # Train model
    history = detector.train(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        epochs=100,
        batch_size=16,
        learning_rate=0.001
    )
    
    # Save model
    detector.save_model("models/face_detection_model.h5")
    
    # Test on validation set
    val_images_dir = "datasets/face detection/images/val"
    val_labels_dir = "datasets/face detection/labels/val"
    
    logger.info("Evaluating on validation set...")
    X_val, _, y_val, _ = detector.load_data(val_images_dir, val_labels_dir, validation_split=0.0)
    
    if len(X_val) > 0:
        val_loss = detector.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Loss: {val_loss}")
    
    return detector

if __name__ == "__main__":
    train_face_detection_model() 