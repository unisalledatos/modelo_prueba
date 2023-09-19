import pickle
import streamlit as st
import numpy as np

st.title("Modelo")

st.divider()

st.write("Ingrese los datos")

educ = st.slider("Educación", 0, 20)
exper = st.slider("Experiencia", 0, 50)
gender = st.selectbox("Gender", [0,1]) 
with open("model.pickle", 'rb') as doc:
    model = pickle.load(doc)

print(type(model))

prediccion = model.predict(np.array([[educ, exper, gender]]))
if st.button("Predecir:"):
    st.write(f"El salario es de {prediccion[0]}")