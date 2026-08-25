import streamlit as st
import pandas as pd

#from data.data import
#from models.model import logistic_Cardio_Predict
# app/logistic.py
from models.model import cardiopredict as logistic_Cardio_Predict
import matplotlib.pyplot as plt
import seaborn as sns
import requests
st.header('Cardiovascular Disease Prediction')
st.subheader('Using Logistic Regression')

features, scaler, model, Y_pred, cr, cm = logistic_Cardio_Predict()

#age = st.text_input('Age', placeholder='Enter your age')
API_URL = 'http://127.0.0.1:8000/predict-cardio-logistic'
st.sidebar.header(
   'Cardio Features'
)

age = st.sidebar.slider(
    'Age',
    max_value=70,
    min_value=26,
    value=30,
    step=1
)

#gender = st.sidebar.slider(
  #  'Gender',
  #  1,2
gender_dict = {
    1: 'Male',
    2: 'Female'
}
gender = st.sidebar.radio(
    'Gender',
    #get keys from the dictionary
    options = list(gender_dict.keys()),
    #get values from the dictionary
    format_func = lambda x: gender_dict.get(x)
)
selected_gender = st.sidebar.selectbox(
    'Gender',
    options = list(gender_dict.keys()),
    format_func = lambda x: gender_dict.get(x)
)
#gender = gender_dict[selected_gender]
gender = selected_gender


height = st.sidebar.slider(
    'Height',
    max_value=200,
    min_value=136,
    value=145,
    step=1
)

weight = st.sidebar.slider(
    'Weight',
    max_value=120,
    min_value=35,
    value=60,
    step=1
)

ap_hi = st.sidebar.slider(
    'Systolic Pressure',
    max_value=200,
    min_value=90,
    value=120,
    step=1
)

ap_lo = st.sidebar.slider(
    'Dy-Systolic Pressure',
    max_value=100,
    min_value=50,
    value=80,
    step=1
)

cholestrol_dict = {
    1: 'Low Cholestrol',
    2: 'Mild Cholestrol', 
    3: 'High Cholestrol'
}
cholestrol = st.sidebar.radio(
    'Cholestrol',
    options = list(cholestrol_dict.keys()),
    format_func = lambda x: cholestrol_dict.get(x)
)  
gluc_dict = {
    1: 'Low Glucose',
    2: 'Mild Glucose',
    3: 'High Glucose'   
}
gluc = st.sidebar.selectbox(
    'Glucose',
    options = list(gluc_dict.keys()),
    format_func = lambda x: gluc_dict.get(x)
)  

smoke_dict = {0: 'Doesnot Smoke', 1: 'Does Smoke'}
smoke = st.sidebar.selectbox(
    'Smoke',
    options = list(smoke_dict.keys()),
    #get values
    format_func = lambda x: smoke_dict.get(x)
)
alco_dict = {0: 'Doesnot Drink Alcohol', 1: 'Does Drink Alcohol'}
alco = st.sidebar.selectbox(
    'Alcohol',
    options = list(alco_dict.keys()),
    format_func = lambda x: alco_dict.get(x)
)
active_dict = {0: 'Doesnt do PA', 1: 'Does do PA'}
active = st.sidebar.selectbox(
    'Physical Activity',
    options = list(active_dict.keys()),
    #get values
    format_func = lambda x: active_dict.get(x)
)

# if st.button('Predict cardio'):
#     input_data = pd.DataFrame(
#         [[age, gender, height, weight, ap_hi, ap_lo, cholestrol, gluc, smoke, alco, active]],
#         columns=features)
#         #Data scaling
#     input_scalar = scaler.transform(input_data)
#         #predict using model
#     prediction = model.predict(input_scalar)[0]
        
#         #show answer
#     if prediction == 0:
#             st.write ('likely not to have cardiovascular disease')
#             st.success('You are not at risk of Cardiovascular Disease') 
#     else:
#             st.write('You are at risk of Cardiovascular Disease') 
#             st.warning('cardiovascular disease is found.')
if st.button('Predict cardio'):
    payload = {
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholestrol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active
    }
    try:
        response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result['prediction'] == 0:
                st.write('Likely to be healthy.')
                st.success('You are not at risk of Cardiovascular Disease')
            else:
                st.write('Likely to be unhealthy.')
                st.warning('You are at risk of Cardiovascular Disease')
        else:
            st.error(f'API Status Code Error: {response.status_code}')
    except requests.exceptions.RequestException as e:
        st.error(f'API Server Error: {e}')
#visualization
st .subheader('Visualization')    
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='1.0f', xticklabels=['predicted Healthy[0]', 'predicted Unhealthy[1]'], yticklabels=['actual Healthy[0]', 'Actual Unhealthy[1]'])
plt.title('Actual cardio vs. predicted cardio')
st.pyplot(fig)




