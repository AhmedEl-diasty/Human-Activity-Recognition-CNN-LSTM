import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

def create_model(input_shape, num_classes):
    """Create a CNN-LSTM model"""
    model = Sequential([
        # CNN layers
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.25),
        
        # LSTM layers
        LSTM(100, return_sequences=True),
        Dropout(0.25),
        LSTM(50),
        Dropout(0.25),
        
        # Classification layers
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, output_dir):
    """Plot training curves"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot accuracy curve
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss curve
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, 'plots', 'confusion_matrix.png'))
    plt.close()

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """Train the model"""
    # Always use the same output directory
    output_dir = os.path.join('output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    tensorboard = TensorBoard(
        log_dir=os.path.join(output_dir, 'logs'),
        histogram_freq=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, tensorboard]
    )
    
    return history, output_dir

def evaluate_model(model, X_test, y_test, output_dir):
    """Evaluate the model"""
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred_classes, output_dir)
    
    return test_loss, test_accuracy

def predict_activity(model, data):
    """Predict activity from input data"""
    predictions = model.predict(data)
    return np.argmax(predictions, axis=1)

if __name__ == "__main__":
    # Load data
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data', 'processed'))
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))
    model = create_model(input_shape, num_classes)

    # Save model architecture image
    os.makedirs(os.path.join('docs', 'images'), exist_ok=True)
    plot_model(model, to_file=os.path.join('docs', 'images', 'model_architecture.png'), show_shapes=True, show_layer_names=True)
    
    # Train model
    history, output_dir = train_model(model, X_train, y_train, X_test, y_test)
    
    # Plot training curves
    plot_training_history(history, output_dir)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, output_dir)
    
    print(f"\nResults saved in: {output_dir}")
    print("You can use the trained model for prediction using the predict_activity() function") 