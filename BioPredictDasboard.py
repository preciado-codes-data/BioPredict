import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def datos():
    data = pd.read_csv('Cancer_Hueso.csv')
    return data

def visualizacion(data):
    # Visualización de Datos
    pairplot = sns.pairplot(data, hue="Cancer De Hueso")
    st.pyplot(pairplot.fig)

def EDA(data):
    df = pd.DataFrame(data.describe())

    return df

def presion_Arterial(data):
    medias_presion = data.groupby('Cancer De Hueso')['Presion Arterial'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Cancer De Hueso', y='Presion Arterial', data=medias_presion, color='red')
    ax.set_title('Presion Arterial Por Grupo')
    ax.set_xlabel('Cancer De Hueso')
    ax.set_ylabel('Media De Presion Arterial')
    st.pyplot(fig)

def colesterol(data):
    medias_coles = data.groupby('Cancer De Hueso')['Colesterol'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Cancer De Hueso', y='Colesterol', data=medias_coles)
    ax.set_title('Colesterol Por Grupo')
    ax.set_xlabel('Cancer De Hueso')
    ax.set_ylabel('Media Del Colesterol')
    st.pyplot(fig)

def peso(data):
    medias_peso = data.groupby('Cancer De Hueso')['Peso'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Cancer De Hueso', y='Peso', data=medias_peso, color='yellow')
    ax.set_title('Peso Por Grupo')
    ax.set_xlabel('Cancer De Hueso')
    ax.set_ylabel('Media De Pesos')
    st.pyplot(fig)

def altura(data):
    medias_altura = data.groupby('Cancer De Hueso')['Altura'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Cancer De Hueso', y='Altura', data=medias_altura, color='green')
    ax.set_title('Altura Por Grupo')
    ax.set_xlabel('Cancer De Hueso')
    ax.set_ylabel('Media De Alturas')
    st.pyplot(fig)

def edades(data):
    medias_edad = data.groupby('Cancer De Hueso')['Edad'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Cancer De Hueso', y='Edad', data=medias_edad, color='silver')
    ax.set_title('Edades Por Grupo')
    ax.set_xlabel('Cancer De Hueso')
    ax.set_ylabel('Media De Edades')
    st.pyplot(fig)

def modelo_predictor(data, input_data):
    # Variables predictoras y variable objetivo
    X = data.drop(columns=["Cancer De Hueso"])
    y = data["Cancer De Hueso"]

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regresión Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    classific = classification_report(y_test, y_pred) 

    df_nuevo = pd.DataFrame(input_data)

    # Realizar predicciones
    predicciones = model.predict(df_nuevo)

    # Obtener probabilidades de predicción
    probabilidades = model.predict_proba(df_nuevo)[:, 1]  # Probabilidad de pertenecer a la clase positiva (cáncer de hueso)

    return classific, accuracy, probabilidades, predicciones

# Función para recoger las entradas del usuario
def input_usuario():
    Edad = st.number_input("Edad", min_value=0, max_value=100, value=53)
    Sexo = st.selectbox("Sexo", options=[0, 1], format_func=lambda x: "Hombre" if x == 0 else "Mujer")
    Altura = st.number_input("Altura (cm)", min_value=0.0, max_value=250.0, value=166.09)
    Peso = st.number_input("Peso (kg)", min_value=0.0, max_value=200.0, value=71.74)
    Presion_Arterial = st.number_input("Presion Arterial", min_value=0.0, max_value=200.0, value=112.32)
    Colesterol = st.number_input("Colesterol", min_value=0.0, max_value=400.0, value=265.30)

    input_data = {
        "Edad": [Edad],
        "Sexo": [Sexo],
        "Altura": [Altura],
        "Peso": [Peso],
        "Presion Arterial": [Presion_Arterial],
        "Colesterol": [Colesterol]
    }

    return input_data

def grafico(data, input_data):
    # Variables predictoras y variable objetivo
    X = data.drop(columns=["Cancer De Hueso", "Tipo"], errors='ignore')
    y = data["Cancer De Hueso"]

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regresión Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    df_nuevo = pd.DataFrame(input_data)

    df_nuevo["Cancer De Hueso"] = model.predict(df_nuevo)
    data['Tipo'] = 'Inicial'
    df_nuevo['Tipo'] = 'Nuevo'

    df_combinado = pd.concat([data, df_nuevo], ignore_index=True)

    fig = px.scatter(df_combinado, x='Edad', y='Colesterol', color='Cancer De Hueso', symbol='Tipo', 
                    title='Datos Y Predicciones', width=1500, height=800)

    st.plotly_chart(fig, use_container_width=True)

def grafico_dispersion(data, input_data):
     # Variables predictoras y variable objetivo
    X = data.drop(columns=["Cancer De Hueso"])
    y = data["Cancer De Hueso"]

    # División de datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo de Regresión Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    df_nuevo = pd.DataFrame(input_data)

    df_nuevo["Cancer De Hueso"] = model.predict(df_nuevo)
    data['Tipo'] = 'Inicial'
    df_nuevo['Tipo'] = 'Nuevo'

    df_combinado = pd.concat([data, df_nuevo], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for cancer in [0,1]:
        subset = df_combinado[(df_combinado['Tipo'] == 'Inicial') & (df_combinado['Cancer De Hueso'] == cancer)]
        ax.scatter(subset['Edad'], subset['Colesterol'], label=f'Inicial - Cancer {cancer}', alpha=0.7)

    for cancer in [0,1]:
        subset = df_combinado[(df_combinado['Tipo'] == 'Nuevo') & (df_combinado['Cancer De Hueso'] == cancer)]
        ax.scatter(subset['Edad'], subset['Colesterol'], label=f'Nuevo - Cancer {cancer}', edgecolor='black', alpha=0.7, marker='x' if cancer == 1 else 'o')

    ax.set_xlabel('Edad')
    ax.set_ylabel('Colesterol')
    ax.legend()
    ax.set_title('Datos Y Predicciones')
    st.pyplot(fig)

def correlacion(data):
    # Calcular la matriz de correlación
    matrix = data.corr()
    
    # Crear un mapa de calor
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlación')
    
    # Mostrar el mapa de calor en Streamlit
    st.pyplot(plt.gcf())

if __name__ == "__main__":
    # Configuración de la página
    st.set_page_config(
        page_title="BioPredict Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    date = datos()

    st.balloons()
    st.title("Bio Predict")
    st.header("Bio Predict Dashboard")
    st.subheader("Somos una empresa enfocada al analisis y prevencion del cancer de hueso")
    st.subheader("Dashboard Interactivo de nuestros servicios en Bio Predict")
    st.caption("Desarrollado Por: ")
    st.caption("Victor Jesus Martinez Perez")
    st.caption("Jose Ramon Preciado Torres")
    st.caption("Gabriel Alejandro Gudiño Mendez")
    st.caption("Dantar Alejandro Ortiz Vega")

    opcion = st.sidebar.selectbox("Selecciona Una Opcion", ["Analisis Exploratorio De Datos (EDA)","Visualizacion De Patrones Del Cancer De Hueso", "Predictor De Cancer De Hueso"])

    if opcion == 'Analisis Exploratorio De Datos (EDA)':
        st.sidebar.header("Analisis Inicial Exploratorio De Los Datos")
        st.sidebar.caption("""En esta seccion del dashboard ofrecemos un servicio inicial que es super importante realizar el cual es un Analisis Inicial de los
                           datos en el cual se realiza una descripcion de medias aritmeticas y matematicas de cada una de las variables numericad del dataset""")
        st.header('Analisis EDA')
        st.write(EDA(date))
        st.write(correlacion(date))

    if opcion == 'Visualizacion De Patrones Del Cancer De Hueso':
        st.sidebar.header("Visualizaciones De Patrones y Relaciones")
        st.sidebar.caption("""En esta seccion del dashboard ofrecemos servicios de analisis y visualizacion de los datos para asi poder encontrar patrones que nos
                           ayude a poder entender los factores de contagio del cancer de hueso""")
        
        st.header('Visualizacion General Para Encontrar Analisis De Patrones Del Cancer De Hueso')
        st.write(visualizacion(date))
        st.header('Analisis De Patrones Del Cancer De Hueso')
        st.write(presion_Arterial(date))
        st.write(colesterol(date))
        st.write(peso(date))
        st.write(altura(date))
        st.write(edades(date))
    
    if opcion == 'Predictor De Cancer De Hueso':
        st.sidebar.header("Modelo Predictor")
        st.sidebar.caption("""En esta seccion del dashboard ofrecemos el servicio del modelo predictor de un nuevo paciente en base a unas metricas como colesterol
                           presion arterial entre otras""")
        # Pedir las entradas del usuario
        st.write("## Introduce los datos del paciente")
        input_data = input_usuario()

        # Botón para calcular las predicciones
        if st.button("Calcular"):
            clasificacion, efectividad, proba, predic = modelo_predictor(date, input_data)

            st.header('Modelo Predictor')
            st.write(f"El Modelo Tiene Una Efectividad del: {efectividad*100}%")
            for x in proba:
                st.write(f"El paciente tiene una probabilidad de {x:.2f}% de pertenecer a la clase: {predic}")
            
            st.write(grafico_dispersion(date, input_data))
            st.write(grafico(date, input_data))

    # Agregar un pie de página
    st.markdown("""
        ---
        **BioPredict** - Todos los derechos reservados.
        Contacto: [info@biopredict.com](mailto:info@biopredict.com)
    """)