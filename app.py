import streamlit as st
import joblib
import numpy as np

# Cargar el modelo
modelo = joblib.load("iris_model.joblib")

# Título
st.title("Clasificador de Flores – Iris")

# Inputs
sl = st.slider("Largo del sepalo (cm)", 4.0, 8.0, 5.0)
sw = st.slider("Ancho del sepalo (cm)", 2.0, 4.5, 3.0)
pl = st.slider("Largo del petalo (cm)", 1.0, 7.0, 4.0)
pw = st.slider("Ancho del petalo (cm)", 0.1, 2.5, 1.0)

# Predicción
if st.button("Predecir tipo de flor"):
    datos = np.array([[sl, sw, pl, pw]])
    pred = modelo.predict(datos)
    nombres = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Prediccion: {nombres[pred[0]]}")
