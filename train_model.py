import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Cardiovascular_Disease.csv"
LOGISTIC_MODEL_PATH = BASE_DIR / "models" / "logistic" / "logistic_model.pkl"
LOGISTIC_SCALER_PATH = BASE_DIR / "models" / "logistic" / "logistic_scaler.pkl"

df = pd.read_csv(DATA_PATH)

# Data Cleaning
df['age'] = df['age'] // 365

df = df[
    (df['height'].between(136, 200)) &
    (df['weight'].between(35, 120)) &
    (df['ap_hi'].between(90, 200)) &
    (df['ap_lo'].between(50, 100))
]

def cardiopredict():
    features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target = 'cardio'
    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train)
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver='lbfgs',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_scale, Y_train)

    LOGISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LOGISTIC_MODEL_PATH)
    joblib.dump(scaler, LOGISTIC_SCALER_PATH)

    print("Model saved to:", LOGISTIC_MODEL_PATH)
    print("Scaler saved to:", LOGISTIC_SCALER_PATH)

if __name__ == "__main__":
    cardiopredict()