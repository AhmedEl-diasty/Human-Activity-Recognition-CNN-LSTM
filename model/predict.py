import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_trained_model(model_path):
    """Load the trained model"""
    return load_model(model_path)

def predict_activity(model, data):
    """Predict activity from input data"""
    predictions = model.predict(data)
    return np.argmax(predictions, axis=1)

def plot_prediction(data, prediction, activity_names):
    """Plot input data and prediction probabilities"""
    plt.figure(figsize=(12, 6))
    
    # Plot input data
    plt.subplot(1, 2, 1)
    plt.plot(data[0, :, 0], label='Feature 1')
    plt.plot(data[0, :, 1], label='Feature 2')
    plt.title('Input Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    
    # Plot prediction probabilities
    plt.subplot(1, 2, 2)
    plt.bar(activity_names, prediction[0])
    plt.title('Activity Probabilities')
    plt.xlabel('Activity')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the trained model
    model_path = os.path.join('output', '20250510_183201', 'best_model.h5')
    model = load_trained_model(model_path)
    
    # List of activity names
    activity_names = [
        'Walking',
        'Walking Upstairs',
        'Walking Downstairs',
        'Sitting',
        'Standing'
    ]
    
    # Example prediction
    # Replace this with your own data
    sample_data = np.random.randn(1, 128, 561)  # Shape: (1, window_size, features)
    
    # Prediction
    prediction = model.predict(sample_data)
    predicted_class = np.argmax(prediction, axis=1)
    
    print(f"Predicted activity: {activity_names[predicted_class[0]]}")
    print("\nActivity probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"{activity_names[i]}: {prob:.4f}")
    
    # Plot results
    plot_prediction(sample_data, prediction, activity_names) 