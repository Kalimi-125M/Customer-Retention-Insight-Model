%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained KNN model
model = joblib.load('churn_model.pkl')

# Load the preprocessors
le_geography = joblib.load('label_encoder_geography.pkl')
le_gender = joblib.load('label_encoder_gender.pkl')
scaler = joblib.load('min_max_scaler.pkl')

st.title('Customer Churn Prediction')
st.write('Enter customer details to predict churn.')

# Create input fields for user
credit_score = st.slider('CreditScore', 350, 850, 600)
geography = st.selectbox('Geography', le_geography.classes_)
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92, 35)
tenure = st.slider('Tenure (years)', 0, 10, 5)
bat_balance = st.number_input('Balance', 0.0, 250000.0, 50000.0)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 60000.0)

if st.button('Predict Churn'):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': bat_balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member,
        'EstimatedSalary': estimated_salary
    }])

    # Apply label encoding
    input_data['Geography'] = le_geography.transform(input_data['Geography'])
    input_data['Gender'] = le_gender.transform(input_data['Gender'])

    # Apply Min-Max scaling for numerical features
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'The customer is likely to churn with a probability of {prediction_proba[0]:.2f}')
    else:
        st.success(f'The customer is unlikely to churn with a probability of {prediction_proba[0]:.2f}')

st.markdown("""
### Files to upload to GitHub:
- `app.py`
- `requirements.txt`
- `churn_model.pkl`
- `label_encoder_geography.pkl`
- `label_encoder_gender.pkl`
- `min_max_scaler.pkl`
""")