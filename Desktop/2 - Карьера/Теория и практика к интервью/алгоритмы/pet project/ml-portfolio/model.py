# model.py (КОПИРУЙ ВСЮ КОД)
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np

model = None

def create_titanic_data():
    np.random.seed(42)
    n = 1000
    data = {
        'pclass': np.random.choice([1,2,3], n, p=[0.24,0.22,0.54]),
        'sex': np.random.choice([0,1], n, p=[0.65,0.35]),
        'age': np.random.normal(30, 15, n).clip(1,80),
        'sibsp': np.random.poisson(0.5, n).clip(0,8),
        'parch': np.random.poisson(0.4, n).clip(0,6),
        'fare': np.random.exponential(15, n).clip(0,500)
    }
    df = pd.DataFrame(data)
    df['survived'] = (
        (df['sex'] == 1) | (df['pclass'] == 1) | (df['age'] < 15)
    ).astype(int) + np.random.normal(0, 0.3, n)
    df['survived'] = (df['survived'] > 0.5).astype(int)
    return df

def predict_survival(data):
    global model
    if model is None:
        print("🎓 Обучаем модель...")
        df = create_titanic_data()
        X = df.drop('survived', axis=1)
        y = df['survived']
        model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'titanic_model.pkl')
        print("✅ Готово!")
    
    df = pd.DataFrame(data, columns=['pclass','sex','age','sibsp','parch','fare'])
    return model.predict_proba(df)[:, 1].tolist()
