import os
import urllib.request
import zipfile

def download_uci_har_dataset():
    """تحميل مجموعة بيانات UCI HAR"""
    # إنشاء مجلد البيانات إذا لم يكن موجوداً
    os.makedirs('../data/raw', exist_ok=True)
    
    # رابط تحميل البيانات
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    zip_path = "../data/raw/UCI_HAR_Dataset.zip"
    
    try:
        # تحميل الملف
        print("جاري تحميل البيانات...")
        urllib.request.urlretrieve(url, zip_path)
        
        # فك ضغط الملف
        print("جاري فك ضغط البيانات...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("../data/raw")
        
        # حذف ملف ZIP بعد فك الضغط
        os.remove(zip_path)
        print("تم تحميل البيانات بنجاح!")
        
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل البيانات: {e}")

if __name__ == "__main__":
    download_uci_har_dataset() 