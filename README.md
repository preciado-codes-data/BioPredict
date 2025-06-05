# 🔬 BioPredict Dashboard  

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python" alt="Python version">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange" alt="scikit-learn">
  <br>
  <em>Herramienta interactiva para análisis y predicción de riesgo de cáncer de hueso</em>
</div>

---

**BioPredict** es una herramienta interactiva basada en Streamlit que permite analizar, visualizar y predecir la probabilidad de padecer **cáncer de hueso** a partir de variables biométricas como la edad, sexo, presión arterial, colesterol, peso y altura.

## 📊 Descripción del Proyecto

Esta aplicación está enfocada en tres funcionalidades principales:

1. **Análisis Exploratorio de Datos (EDA)**  
   Descripción estadística básica de las variables numéricas del conjunto de datos.

2. **Visualización de Patrones del Cáncer de Hueso**  
   Gráficos y visualizaciones interactivas para descubrir patrones relevantes.

3. **Modelo Predictor de Cáncer de Hueso**  
   Clasificación mediante regresión logística, permitiendo ingresar datos de un nuevo paciente y predecir su probabilidad de pertenecer a la clase con o sin cáncer.

---

## 🧠 Tecnologías Utilizadas

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- Streamlit

---

## 📁 Estructura del Proyecto

```
BioPredictDashboard/
  ├── BioPredictDasboard.py # Archivo principal con toda la lógica del dashboard
  ├── Cancer_Hueso.csv # Dataset utilizado
  ├── README.md # Este archivo
```

---

## ▶️ Cómo Ejecutar la Aplicación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu_usuario/BioPredictDashboard.git
   cd BioPredictDashboard

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt

3. Ejecuta la aplicación en tu navegador:
   ```bash
   streamlit run BioPredictDasboard.py

---

## 👨‍💻 Desarrolladores

- Víctor Jesús Martínez Pérez
- José Ramón Preciado Torres

---

## ⚠️ Notas

- El modelo de predicción está basado en un clasificador de regresión logística entrenado sobre datos proporcionados en Cancer_Hueso.csv.
- Este proyecto tiene fines educativos y de demostración.
