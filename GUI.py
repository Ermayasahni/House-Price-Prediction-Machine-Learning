import os 
print("Current working directory:", os.getcwd())

import streamlit as st
import pandas as pd
import pickle


model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 'XGBoost', 'KNN'
]

models = {name: pickle.load(open(f"{name}.pkl", "rb")) for name in model_names}

results_df = pd.read_csv("model_performance.csv")


st.title("House Price Prediction App")

menu = st.sidebar.radio("Navigation", ["Predict", "Model Performance"])

if menu == "Predict":
    st.header("🏠 Predict House Price")

    # Model selection
    model_selected = st.selectbox("Select Model", model_names)

    # Input fields
    income = st.number_input("Avg. Area Income", min_value=0.0)
    age = st.number_input("Avg. Area House Age", min_value=0.0)
    rooms = st.number_input("Avg. Area Number of Rooms", min_value=0.0)
    bedrooms = st.number_input("Avg. Area Number of Bedrooms", min_value=0.0)
    population = st.number_input("Area Population", min_value=0.0)

    # Predict button
    if st.button("Predict Price"):
        input_data = {
            'Avg. Area Income': income,
            'Avg. Area House Age': age,
            'Avg. Area Number of Rooms': rooms,
            'Avg. Area Number of Bedrooms': bedrooms,
            'Area Population': population
        }

        input_df = pd.DataFrame([input_data])
        model = models[model_selected]

        prediction = model.predict(input_df)[0]            

        st.success(f"🏡 **Predicted Price:** ${prediction:,.2f} using {model_selected}")


elif menu == "Model Performance":
    st.header("📊 Model Performance Table")

    st.write("Below is the model performance comparison:")
    st.dataframe(results_df)
