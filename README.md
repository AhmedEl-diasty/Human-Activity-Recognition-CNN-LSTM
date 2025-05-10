# Human Activity Recognition using CNN-LSTM

This project implements a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for human activity recognition using sensor data from smartphones.

## 🎯 Features

- Hybrid CNN-LSTM architecture for robust activity recognition
- Real-time activity prediction capability
- Comprehensive model evaluation and visualization
- Support for UCI HAR Dataset
- Detailed performance metrics and visualizations

## 📊 Model Architecture

The model combines:
- CNN layers for spatial feature extraction
- LSTM layers for temporal pattern recognition
- Dense layers for final classification

![Model Architecture](output/plots/model_architecture.png)

## 📁 Project Structure

```
├── data/
│   ├── raw/          # Original UCI HAR Dataset
│   └── processed/    # Preprocessed data ready for training
├── model/
│   ├── cnn_lstm.py   # CNN-LSTM model implementation
│   └── predict.py    # Prediction script
├── notebooks/
│   └── har_cnn_lstm.ipynb  # Jupyter notebook for interactive development
├── output/
│   ├── best_model.h5       # Trained model
│   └── plots/             # Training and evaluation plots
└── utils/
    ├── preprocess.py      # Data preprocessing utilities
    └── download_data.py   # Dataset download script
```

## 🛠️ Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- OpenCV

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Getting Started

1. Download the dataset:
```bash
python utils/download_data.py
```

2. Preprocess the data:
```bash
python utils/preprocess.py
```

3. Train the model:
```bash
python model/cnn_lstm.py
```

4. Make predictions:
```bash
python model/predict.py
```

## 📈 Results

### Training Performance

![Training Curves](output/plots/training_history.png)

### Model Evaluation

![Confusion Matrix](output/plots/confusion_matrix.png)

### Classification Report
```
              precision    recall  f1-score   support

    Walking       0.50      0.27      0.35        11
Walking Up       0.00      0.00      0.00         3
Walking Down     0.00      0.00      0.00         5
    Sitting      0.52      0.92      0.67        12
   Standing      0.50      0.64      0.56        14

    accuracy                           0.51        45
   macro avg      0.30      0.37      0.32        45
weighted avg      0.42      0.51      0.44        45
```

## 🔍 Model Details

### Architecture
- Input shape: (128, 561)
- CNN layers: 2 Conv1D layers with MaxPooling
- LSTM layers: 2 LSTM layers
- Dense layers: 2 Dense layers with Dropout
- Output: 5 activity classes

### Activities Recognized
1. Walking
2. Walking Upstairs
3. Walking Downstairs
4. Sitting
5. Standing

## 📝 Notes

- The model uses the UCI HAR Dataset
- Training history and model checkpoints are saved in the output directory
- The best model is saved as 'best_model.h5'
- Performance metrics and visualizations are automatically generated

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
