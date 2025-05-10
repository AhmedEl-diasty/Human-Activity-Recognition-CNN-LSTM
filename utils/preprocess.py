import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(data_dir):
    """تحميل البيانات من مجلد UCI HAR Dataset"""
    try:
        # تحميل بيانات التدريب
        X_train = pd.read_csv(os.path.join(data_dir, 'train', 'X_train.txt'), sep='\s+', header=None)
        y_train = pd.read_csv(os.path.join(data_dir, 'train', 'y_train.txt'), header=None)[0]
        
        # تحميل بيانات الاختبار
        X_test = pd.read_csv(os.path.join(data_dir, 'test', 'X_test.txt'), sep='\s+', header=None)
        y_test = pd.read_csv(os.path.join(data_dir, 'test', 'y_test.txt'), header=None)[0]
        
        # تحميل أسماء الميزات
        features = pd.read_csv(os.path.join(data_dir, 'features.txt'), sep='\s+', header=None)[1]
        X_train.columns = features
        X_test.columns = features
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"خطأ في تحميل البيانات: {e}")
        return None, None, None, None

def create_sliding_windows(X, y, window_size=128, step=64):
    """تقسيم البيانات إلى نوافذ زمنية"""
    X_windows = []
    y_windows = []
    for i in range(0, len(X) - window_size + 1, step):
        X_windows.append(X[i:i+window_size].values)
        # نأخذ التصنيف الأكثر تكراراً في النافذة
        y_windows.append(y[i:i+window_size].mode()[0])
    return np.array(X_windows), np.array(y_windows)

def preprocess_data(X_train, X_test, y_train, y_test, window_size=128, step=64):
    """معالجة البيانات وتطبيعها وتقسيمها إلى نوافذ زمنية"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    # تقسيم إلى نوافذ
    X_train_win, y_train_win = create_sliding_windows(X_train_scaled, y_train, window_size, step)
    X_test_win, y_test_win = create_sliding_windows(X_test_scaled, y_test, window_size, step)
    # تطبيع التسميات
    label_encoder = LabelEncoder()
    y_train_win = label_encoder.fit_transform(y_train_win)
    y_test_win = label_encoder.transform(y_test_win)
    return X_train_win, X_test_win, y_train_win, y_test_win, scaler

def save_processed_data(X_train, X_test, y_train, y_test, output_path):
    """حفظ البيانات المعالجة"""
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'X_train.npy'), X_train)
    np.save(os.path.join(output_path, 'X_test.npy'), X_test)
    np.save(os.path.join(output_path, 'y_train.npy'), y_train)
    np.save(os.path.join(output_path, 'y_test.npy'), y_test)

if __name__ == "__main__":
    # المسارات
    data_dir = os.path.join('data', 'raw', 'UCI HAR Dataset')
    output_path = os.path.join('data', 'processed')
    
    # تحميل البيانات
    X_train, X_test, y_train, y_test = load_data(data_dir)
    
    if X_train is not None:
        # معالجة البيانات
        X_train_processed, X_test_processed, y_train, y_test, scaler = preprocess_data(
            X_train, X_test, y_train, y_test, window_size=128, step=64
        )
        
        # حفظ البيانات المعالجة
        save_processed_data(X_train_processed, X_test_processed, y_train, y_test, output_path)
        print("تم معالجة وحفظ البيانات بنجاح!") 