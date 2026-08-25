# ML modelsimport pandas as pd
# ML models
from pathlib import Path
import pandas as pd
import joblib

# Logistic path — resolved relative to this file, not the terminal's cwd
BASE_DIR = Path(__file__).resolve().parent.parent  # Cardio/ (parent of app/)
LOGISTIC_MODEL_PATH = BASE_DIR / "models" / "logistic" / "logistic_model.pkl"
LOGISTIC_SCALER_PATH = BASE_DIR / "models" / "logistic" / "logistic_scaler.pkl"

def load_logistic_model():
    model = joblib.load(LOGISTIC_MODEL_PATH)
    scaler = joblib.load(LOGISTIC_SCALER_PATH)
    return model, scaler


# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# #to convert model into binary format
# import joblib
# df = pd.read_csv('Data/Cardiovascular_Disease.csv')

# #Data Cleaning
# df['age'] = df['age']//365

# hw = df[
# (df['height'].between(136, 200)) &
# (df['weight'].between(35, 120)) &
# (df['ap_hi'].between(90, 200)) &
# (df['ap_lo'].between(50, 100))
# ]
# df = hw
# #Logistic Path
# LOGISTIC_MODEL_PATH = 'models/logistic_model.pkl'
# LOGISTIC_SCALER_PATH = 'models/logistic_scaler.pkl'

# def cardiopredict():
#     features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
#     target = 'cardio'
#     X = df[features]
#     Y = df[target]
#     X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=0.2, random_state=42, stratify=Y
#     )
#     # Xi - mean / sd
#     scaler = StandardScaler()
#     X_train_scale = scaler.fit_transform(X_train) # learning + implement
#     X_test_scale = scaler.transform(X_test) # implement

#     # solver -> liblinear, lbfgs, saga
#     model = LogisticRegression(
#     solver='lbfgs',
#     class_weight='balanced',
#     random_state=42
#     )
#     model.fit(X_train_scale, Y_train)
#     model.predict(X_test_scale)
   
#     Joblib.dump(model, LOGISTIC_MODEL_PATH)
#     joblib.dump(scaler, LOGISTIC_SCALER_PATH) 
  
#     return scaler, model

