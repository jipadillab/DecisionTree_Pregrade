import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración inicial de la página
st.set_page_config(page_title="Clase IA - Árboles y Ensambles", layout="wide")

def plot_decision_boundary(model, X, y, title):
    """Función auxiliar para graficar fronteras de decisión"""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    ax.set_title(title)
    return fig

# Generación de datos de prueba
@st.cache_data
def load_data(dataset_type):
    if dataset_type == "Moons (Complejo)":
        X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
    else:
        X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                   n_clusters_per_class=1, weights=[0.9, 0.1], random_state=42) # Clases desbalanceadas
    return X, y

# Menú lateral
st.sidebar.image("https://www.eafit.edu.co/Style%20Library/Eafit/images/logo-eafit-blanco.png", width=150)
st.sidebar.title("Contenido de la Clase")
menu = st.sidebar.radio("Selecciona un tema:", 
                        ["1. Introducción", 
                         "2. Decision Trees", 
                         "3. Bagging (Bootstrap Aggregating)", 
                         "4. Boosting", 
                         "5. Kaggle Context"])

X, y = load_data("Moons (Complejo)")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if menu == "1. Introducción":
    st.header("1. Introducción a los Modelos y Ensambles")
    st.markdown("""
    Los modelos simples son eficientes y de rápido entrenamiento, pero pueden llegar a tener poco poder explicativo sobre datos observados (**subajuste**)[cite: 25]. 
    Por otro lado, los modelos muy complejos tienen mucho poder explicativo sobre un conjunto de entrenamiento pero poca habilidad predictiva para datos no observados (**sobreajuste**)[cite: 26].
    
    Además, en problemas reales nos enfrentamos a **clases no balanceadas**[cite: 30], como en:
    * Diagnósticos médicos [cite: 34]
    * Detección de fraude [cite: 34]
    * Detección de Spam [cite: 35]
    """)
    
    st.subheader("Visualización del problema")
    dataset_choice = st.selectbox("Selecciona un dataset para visualizar:", ["Moons (Complejo)", "Clases Desbalanceadas"])
    X_viz, y_viz = load_data(dataset_choice)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_viz[:,0], y=X_viz[:,1], hue=y_viz, palette="coolwarm", ax=ax)
    st.pyplot(fig)

elif menu == "2. Decision Trees":
    st.header("2. Árboles de Decisión (Decision Trees)")
    st.markdown("""
    Un árbol general consta de un nodo raíz (*root node*), nodos internos (*split nodes*) y nodos terminales u hojas (*leaf nodes*)[cite: 96, 98, 106].
    
    **¿Cómo se toman las mejores preguntas para dividir los datos?** [cite: 292]
    A través de métricas de impureza como la **Entropía**:
    $H(\pi)=-\sum_{\forall\pi}\pi~log_{2}(\pi)$ [cite: 294]
    
    Y buscando maximizar la **Ganancia de Información (Information Gain)**:
    $I(A)=H(\pi_{p},\pi_{n})-EH(A)$ [cite: 310]
    """)
    
    st.sidebar.markdown("### Hiperparámetros del Árbol")
    max_depth = st.sidebar.slider("Profundidad Máxima (max_depth)", 1, 15, 3)
    criterion = st.sidebar.selectbox("Criterio de División", ["gini", "entropy"])
    
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=42)
    dt.fit(X_train, y_train)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Frontera de Decisión")
        st.pyplot(plot_decision_boundary(dt, X_test, y_test, f"Decision Tree (Depth={max_depth})"))
        st.metric("Accuracy (Test)", f"{accuracy_score(y_test, dt.predict(X_test)):.2f}")
    
    with col2:
        st.subheader("Estructura del Árbol")
        fig_tree, ax_tree = plt.subplots(figsize=(10, 8))
        plot_tree(dt, filled=True, ax=ax_tree, max_depth=2, rounded=True)
        st.pyplot(fig_tree)
        st.caption("Mostrando solo los primeros 2 niveles para claridad visual.")

elif menu == "3. Bagging (Bootstrap Aggregating)":
    st.header("3. Bootstrap Aggregating (Bagging)")
    st.markdown("""
    Bagging toma varios clasificadores simples y entrena cada uno con un subconjunto de los datos [cite: 324] mediante muestreo aleatorio con reemplazo (*bootstrap samples*)[cite: 345, 360].
    
    Las características principales son:
    1. Es un ensamble **paralelo** (entrenados de forma independiente)[cite: 371].
    2. Usualmente es homogéneo[cite: 372].
    3. Un ensamble de Árboles de Decisión con Bagging se llama **Random Forest**[cite: 373].
    """)
    
    st.sidebar.markdown("### Hiperparámetros Random Forest")
    n_estimators = st.sidebar.slider("Número de Árboles (n_estimators)", 1, 100, 10)
    max_depth_rf = st.sidebar.slider("Profundidad de cada Árbol", 1, 10, 3)
    
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_rf, random_state=42)
    rf.fit(X_train, y_train)
    
    st.subheader("Frontera de Decisión: Random Forest")
    st.pyplot(plot_decision_boundary(rf, X_test, y_test, f"Random Forest ({n_estimators} Trees)"))
    st.metric("Accuracy (Test)", f"{accuracy_score(y_test, rf.predict(X_test)):.2f}")

elif menu == "4. Boosting":
    st.header("4. Boosting")
    st.markdown("""
    Boosting engloba algoritmos que mejoran predicciones de manera **secuencial**[cite: 383]. Para cada nuevo modelo, se le da más peso a los datos que no fueron bien clasificados[cite: 396, 398, 399].
    
    En *Gradient Boosting*, el modelo se entrena para minimizar el residual o gradiente de pérdida:
    $r_{1}=y_{1}-\hat{y}_{l}$ [cite: 404, 405]
    """)
    
    st.sidebar.markdown("### Hiperparámetros de Boosting")
    model_choice = st.sidebar.radio("Algoritmo", ["AdaBoost", "Gradient Boosting"])
    n_est_boost = st.sidebar.slider("Número de estimadores", 1, 100, 50)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    
    if model_choice == "AdaBoost":
        boost_model = AdaBoostClassifier(n_estimators=n_est_boost, learning_rate=learning_rate, random_state=42)
    else:
        boost_model = GradientBoostingClassifier(n_estimators=n_est_boost, learning_rate=learning_rate, random_state=42)
        
    boost_model.fit(X_train, y_train)
    
    st.subheader(f"Frontera de Decisión: {model_choice}")
    st.pyplot(plot_decision_boundary(boost_model, X_test, y_test, f"{model_choice} (LR={learning_rate})"))
    st.metric("Accuracy (Test)", f"{accuracy_score(y_test, boost_model.predict(X_test)):.2f}")

elif menu == "5. Kaggle Context":
    st.header("5. Mini-Contexto: ¿Qué es Kaggle?")
    st.markdown("""
        Kaggle es una plataforma en línea especializada en ciencia de datos y aprendizaje automático[cite: 462].
    
    Tus estudiantes podrán:
    * Acceder a datasets públicos[cite: 463].
    * Participar en competencias de ML[cite: 464].
    * Explorar notebooks de otros usuarios[cite: 465].
    * Obtener feedback de la comunidad[cite: 469].
    """)
    st.info("¡Anima a tus estudiantes a entrar a kaggle.com y participar en competencias de clasificación tabular para aplicar lo visto sobre ensambles!")
