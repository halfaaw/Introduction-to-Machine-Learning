import unittest
import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.file_path = "data/loan_data.csv"
        
    def test_file_exists(self):
        self.assertTrue(os.path.exists(self.file_path), "Dataset loan_data.csv tidak ditemukan!")
        
    def test_model_training_basic(self):
        df = pd.read_csv(self.file_path)
        X = df.drop(columns=['loan_id', 'loan_status'])
        y = df['loan_status']
        
        model = DecisionTreeClassifier()
        model.fit(X, y)
        predictions = model.predict(X)
        
        self.assertEqual(len(predictions), len(y), "Jumlah prediksi harus sama dengan jumlah data target!")

if __name__ == '__main__':
    unittest.main()
