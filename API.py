import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge
from xgboost import XGBRFRegressor
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")

path = "C:/Users/monic/OneDrive/Escritorio/Taller C2/"

prueba = pd.read_csv( path + "prueba_APP.csv",header = 0,sep=";",decimal=",")

path2 = "C:/Users/monic/OneDrive/Escritorio/Taller C2/"

import pickle
with open(path2 + 'dt.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

# Streamlit app
st.title("Predicción del precio de la APP basada en características de usuario")

# Entradas del usuario
dominio = st.selectbox("Seleccione el dominio:", ['gmail', 'Otro', 'hotmail', 'yahoo'])
Tec = st.selectbox("Seleccione el tipo de dispositivo:", ['Smartphone', 'Portatil', 'PC', 'Iphone'])

# Campos numéricos sin valor predeterminado
Avg = st.text_input("Ingrese Avg. Session Length:", value= 33.946241)
Time_App = st.text_input("Ingrese Time on App:", value= 10.983977)
Time_Web = st.text_input("Ingrese Time on Website:", value= 37.951489)
Length = st.text_input("Ingrese Length of Membership:", value= 3.050713)

prueba = prueba.drop(columns=['price', 'Email', 'Address'], errors='ignore')


if st.button("Calcular"):
    try:
        Avg = float(Avg)
        Time_App = float(Time_App)
        Time_Web = float(Time_Web)
        Length = float(Length)
        
        user = pd.DataFrame({'x1':[dominio],'x2': [Tec],
        'x3': [Avg], 'x4': [Time_App], 'x5': [Time_Web], 'x6': [Length]})

        user.columns = prueba.columns
        prueba2 = pd.concat([user,prueba],axis = 0)
        prueba2.index = range(prueba2.shape[0])
        prueba2.head(2)

        df_test = prueba2.copy()
        predictions = predict_model(dt2, data=prueba2)
        predictions.head()

        predictions["price"] = predictions["prediction_label"]
        predictions.head()

        st.markdown(f'La predicción es: {predictions.iloc[0]["price"]}')
    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en todos los campos.")
    

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()


    
