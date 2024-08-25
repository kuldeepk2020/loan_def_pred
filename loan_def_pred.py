import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import requests
from io import BytesIO

# Function to download data from Google Drive
def download_data_from_drive(url):
    r = requests.get(url)
    return pd.read_csv(BytesIO(r.content))

# Downloading data (modify the URLs with your actual Google Drive links)
X_train_url = 'https://drive.google.com/uc?export=download&id=1uD7YZn6MmWfFsrAFThnnp4sPRWNkUjeN'
X_test_url = 'https://drive.google.com/uc?export=download&id=18Ol4o6WqZ_h5aHRFSPJoha7d-ePdWzu9'
y_train_url = 'https://drive.google.com/uc?export=download&id=1Mm_cf1TDXHI4GQaPCVL3vkovuWW2IlzQ'
y_test_url = 'https://drive.google.com/uc?export=download&id=1oek1bmQ8AYH7GxFHPZTHQ0E-GEZyIdRO'

# Load data
@st.cache_data
def load_data():
    X_train = download_data_from_drive(X_train_url)
    X_test = download_data_from_drive(X_test_url)
    y_train = download_data_from_drive(y_train_url)
    y_test = download_data_from_drive(y_test_url)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_data()

# Convert target variables to binary
y_train = y_train['Risk_Flag_1'].apply(lambda x: 1 if x else 0)
y_test = y_test['Risk_Flag_1'].apply(lambda x: 1 if x else 0)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# Model training
st.sidebar.subheader('Model Training')
max_depth = st.sidebar.slider('Max Depth', min_value=3, max_value=10, value=5, step=1)
learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=0.5, value=0.1, step=0.01)
n_estimators = st.sidebar.slider('Number of Estimators', min_value=50, max_value=500, value=100, step=50)

if st.sidebar.button('Train Model'):
    st.session_state.model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators)
    st.session_state.model.fit(X_train, y_train)
    y_pred = st.session_state.model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.session_state.trained = True
    st.write(f'Model Accuracy: {accuracy:.2f}')

    # Display feature importances
    st.subheader('Feature Importances')
    importances = st.session_state.model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    st.write(importance_df.sort_values(by='Importance', ascending=False))

# Predictions
st.sidebar.subheader('Make Predictions')
if st.sidebar.checkbox('Predict Loan Defaulters'):
    if st.session_state.trained:
        st.write('Enter values for prediction:')
        # Example inputs for prediction
        example_input = {
            'No_of_partners': st.number_input('Number of Partners', min_value=1, max_value=10, value=2),
            'Income_annum': st.number_input('Income per Annum', min_value=10000, max_value=10000000, value=500000),
            'Loan_amount_in_INR': st.number_input('Loan Amount in INR', min_value=10000, max_value=100000000, value=1000000),
            'Loan_term_yrs': st.number_input('Loan Term in Years', min_value=1, max_value=30, value=5),
            'Cibil_score': st.number_input('CIBIL Score', min_value=300, max_value=900, value=600),
            'Assets_value': st.number_input('Assets Value', min_value=0, max_value=100000000, value=500000),
            'Status_at_Branch': st.selectbox('Status at Branch', [0, 1]),
            'Any_other_ongoing_loans': st.selectbox('Any other ongoing loans', [0, 1])
        }

        # Convert input to DataFrame
        input_df = pd.DataFrame([example_input])

        # Align feature columns
        expected_columns = X_train.columns.tolist()
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Predict
        if st.session_state.model is not None:
            if st.button('Predict'):
                prediction = st.session_state.model.predict(input_df)[0]
                prediction_label = 'True' if prediction == 1 else 'False'
                st.write(f'Risk Flag (Predicted): {prediction_label}')
        else:
            st.write('Model is not trained yet. Please train the model first.')
    else:
        st.write('Model is not trained yet. Please train the model first.')

# Display raw data (optional)
if st.checkbox('Show Raw Data'):
    st.subheader('Raw Data')
    st.write('Training Data:')
    st.write(X_train.head())
    st.write('Testing Data:')
    st.write(X_test.head())









