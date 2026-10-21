import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

def main():
    print("=== Memulai Alur Machine Learning (Supervised Classification) ===")
    
    # 1. SETUP DIREKTORI OUTPUT
    os.makedirs("outputs", exist_ok=True)
    
    # 2. LOAD DATA
    try:
        df = pd.read_csv("data/loan_data.csv")
    except FileNotFoundError:
        print("Error: File data/loan_data.csv tidak ditemukan.")
        return

    # 3. PREPROCESSING & FEATURE SELECTION
    X = df.drop(columns=['loan_id', 'loan_status'])
    y = df['loan_status']
    
    # 4. TRAIN-TEST SPLIT (75% Training, 25% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 5. MODEL TRAINING (Decision Tree Classifier)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 6. PREDICTION & EVALUATION
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    conf_matrix = confusion_matrix(y_test, y_pred, labels=['Approved', 'Rejected'])
    
    print(f"\nAkurasi Model: {accuracy:.2f}%")
    print("Confusion Matrix:\n", conf_matrix)
    
    # 7. MENYIMPAN LAPORAN EVALUASI
    with open("outputs/metrics_report.txt", "w") as f:
        f.write("=== LAPORAN EVALUASI MACHINE LEARNING ===\n")
        f.write(f"Algoritma Digunakan : Decision Tree Classifier\n")
        f.write(f"Akurasi Pengujian   : {accuracy:.2f}%\n")
        f.write("Confusion Matrix    :\n")
        f.write(str(conf_matrix) + "\n")
        f.write("\nCatatan: Model ini dilatih dari data historis. Keputusan akhir tetap di tangan manusia (human oversight).\n")
        
    print("\nProses selesai. Laporan evaluasi disimpan di 'outputs/metrics_report.txt'.")

if __name__ == "__main__":
    main()
