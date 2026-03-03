"""
🌳 Plataforma Educativa: Árboles de Decisión & Ensambles
Universidad EAFIT - Artificial Intelligence Course
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, KFold, learning_curve, validation_curve, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              roc_curve, precision_recall_curve, mean_squared_error, r2_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification, make_moons, make_circles, load_iris, load_breast_cancer
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌳 ML Trees & Ensembles",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a6b8a 0%, #0d4f6b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .main-header p  { font-size: 0.95rem; margin: 0.3rem 0 0; opacity: 0.85; }

    .concept-box {
        background: #f0f7ff;
        border-left: 5px solid #1a6b8a;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .formula-box {
        background: #fff8e7;
        border: 1.5px solid #f0b429;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-family: monospace;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .success-box {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .metric-card {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .section-title {
        color: #1a6b8a;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid #1a6b8a;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .sidebar-menu-item {
        padding: 0.4rem 0.6rem;
        border-radius: 6px;
        margin: 0.2rem 0;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 ML Trees & Ensembles")
    st.markdown("**Universidad EAFIT**")
    st.markdown("---")

    MODULES = {
        "🏠 Inicio": "home",
        "📊 1. Underfitting & Overfitting": "bias_variance",
        "🌲 2. Árboles de Decisión": "decision_trees",
        "📐 3. Entropía & Information Gain": "entropy",
        "📏 4. Métricas de Desempeño": "metrics",
        "✅ 5. Validación & Cross-Validation": "validation",
        "⚖️ 6. Clases Desbalanceadas": "imbalanced",
        "🌳🌳 7. Bootstrap Aggregating (Bagging)": "bagging",
        "🚀 8. Boosting & Gradient Boosting": "boosting",
        "🏆 9. Comparación de Modelos": "comparison",
        "🔬 10. Laboratorio Libre": "lab",
    }

    selected = st.radio(
        "Selecciona un módulo:",
        list(MODULES.keys()),
        label_visibility="collapsed"
    )
    module = MODULES[selected]

    st.markdown("---")
    st.markdown("### 📚 Referencia rápida")
    with st.expander("Glosario"):
        st.markdown("""
- **Underfitting**: modelo muy simple, alto sesgo  
- **Overfitting**: modelo muy complejo, alta varianza  
- **Entropía**: medida de impureza/desorden  
- **Gini**: otra medida de impureza  
- **Bagging**: modelos en paralelo  
- **Boosting**: modelos en secuencia  
- **Random Forest**: bagging de árboles  
- **SMOTE**: oversampling sintético  
        """)


# ═══════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════

def plot_decision_boundary(model, X, y, title="Frontera de Decisión", ax=None):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu',
                          edgecolors='k', linewidths=0.5, s=40)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return fig


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def gini(p):
    return 1 - p**2 - (1 - p)**2


# ═══════════════════════════════════════════════════════
#  MODULE: HOME
# ═══════════════════════════════════════════════════════

def show_home():
    st.markdown("""
    <div class="main-header">
        <h1>🌳 Árboles de Decisión & Ensambles</h1>
        <p>Plataforma interactiva · Artificial Intelligence · Universidad EAFIT</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Mapa del Curso")

    topics = [
        ("📊", "Underfitting & Overfitting", "Sesgo, varianza y el trade-off fundamental"),
        ("🌲", "Árboles de Decisión", "Estructura, splits, profundidad y poda"),
        ("📐", "Entropía & Info Gain", "Cómo el árbol elige la mejor pregunta"),
        ("📏", "Métricas de Desempeño", "Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("✅", "Validación & CV", "Hold-out, K-Fold, Stratified CV"),
        ("⚖️", "Clases Desbalanceadas", "SMOTE, undersampling, class_weight"),
        ("🌳🌳", "Bagging / Random Forest", "Ensamble paralelo, feature importance"),
        ("🚀", "Boosting", "AdaBoost, Gradient Boosting, XGBoost"),
        ("🏆", "Comparación de Modelos", "Benchmark interactivo con datasets reales"),
        ("🔬", "Laboratorio Libre", "Sube tus datos o experimenta con los ejemplos"),
    ]

    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(topics):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.8rem; text-align:left;">
                <span style="font-size:1.5rem">{icon}</span>
                <strong style="color:#1a6b8a; font-size:1rem"> {title}</strong><br>
                <span style="color:#555; font-size:0.85rem">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="concept-box">
    <b>💡 Cómo usar esta plataforma:</b><br>
    Usa el menú lateral para navegar por cada módulo en orden.
    Cada módulo tiene <b>teoría interactiva</b>, <b>sliders para jugar con parámetros</b>,
    <b>visualizaciones en tiempo real</b> y <b>ejercicios guiados</b>.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MODULE 1: UNDERFITTING / OVERFITTING / BIAS-VARIANCE
# ═══════════════════════════════════════════════════════

def show_bias_variance():
    st.markdown('<div class="section-title">📊 Underfitting, Overfitting & Bias-Variance Tradeoff</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🎮 Interactivo: Árbol & Complejidad", "📉 Curvas de Aprendizaje"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-box">
            <b>🔵 Underfitting (Subajuste)</b><br>
            El modelo es <b>demasiado simple</b> para capturar la estructura de los datos.<br><br>
            • Alto <b>sesgo (bias)</b><br>
            • Baja varianza<br>
            • Error alto en entrenamiento Y en test<br>
            • Causa: modelo poco expresivo, pocas features, árbol muy shallow
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-box">
            <b>🔴 Overfitting (Sobreajuste)</b><br>
            El modelo <b>memoriza el entrenamiento</b> pero no generaliza.<br><br>
            • Bajo sesgo<br>
            • Alta <b>varianza</b><br>
            • Error bajo en entrenamiento, ALTO en test<br>
            • Causa: modelo muy complejo, árbol muy profundo
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        <b>Error Total = Sesgo² + Varianza + Ruido irreducible</b><br><br>
        • <b>Sesgo</b>: error por suposiciones incorrectas del modelo<br>
        • <b>Varianza</b>: sensibilidad del modelo a pequeñas fluctuaciones en los datos<br>
        • Reducir uno generalmente aumenta el otro → <i>tradeoff</i>
        </div>
        """, unsafe_allow_html=True)

        # Bias-Variance curve
        fig, ax = plt.subplots(figsize=(8, 4))
        complexity = np.linspace(1, 10, 100)
        bias2 = 5 * np.exp(-0.4 * complexity)
        variance = 0.05 * np.exp(0.5 * complexity)
        total = bias2 + variance + 0.5
        ax.plot(complexity, bias2, 'b-', linewidth=2.5, label='Sesgo²')
        ax.plot(complexity, variance, 'r-', linewidth=2.5, label='Varianza')
        ax.plot(complexity, total, 'k--', linewidth=2.5, label='Error Total')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Ruido irreducible')
        opt_idx = np.argmin(total)
        ax.axvline(complexity[opt_idx], color='green', linestyle='--', alpha=0.7, label='Complejidad óptima')
        ax.fill_between(complexity[:opt_idx], 0, total[:opt_idx], alpha=0.1, color='blue', label='Zona Underfitting')
        ax.fill_between(complexity[opt_idx:], 0, total[opt_idx:], alpha=0.1, color='red', label='Zona Overfitting')
        ax.set_xlabel('Complejidad del Modelo', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Tradeoff Sesgo-Varianza', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(0, 6)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("### 🎮 Visualiza cómo la profundidad del árbol afecta el ajuste")
        col1, col2 = st.columns([1, 2])
        with col1:
            dataset_type = st.selectbox("Dataset", ["Moons", "Circles", "Linear"])
            noise_level = st.slider("Ruido en los datos", 0.0, 0.5, 0.15, 0.05)
            n_samples = st.slider("Número de muestras", 50, 500, 200, 50)
            max_depth = st.slider("Profundidad máxima del árbol", 1, 15, 3)
            test_size = st.slider("Proporción de Test", 0.1, 0.5, 0.3, 0.05)

        np.random.seed(42)
        if dataset_type == "Moons":
            X, y = make_moons(n_samples=n_samples, noise=noise_level, random_state=42)
        elif dataset_type == "Circles":
            X, y = make_circles(n_samples=n_samples, noise=noise_level, random_state=42)
        else:
            X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                                        n_informative=2, random_state=42, n_clusters_per_class=1)
            X += np.random.randn(*X.shape) * noise_level

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        with col2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_decision_boundary(model, X_train, y_train, f"Train | Acc={train_acc:.2%}", axes[0])
            plot_decision_boundary(model, X_test, y_test, f"Test | Acc={test_acc:.2%}", axes[1])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        col_a, col_b, col_c = st.columns(3)
        diff = train_acc - test_acc
        if diff > 0.1:
            status = "🔴 Posible OVERFITTING"
            status_color = "#dc3545"
        elif test_acc < 0.65:
            status = "🔵 Posible UNDERFITTING"
            status_color = "#007bff"
        else:
            status = "🟢 Buen ajuste"
            status_color = "#28a745"

        col_a.metric("Accuracy Entrenamiento", f"{train_acc:.2%}")
        col_b.metric("Accuracy Test", f"{test_acc:.2%}", delta=f"{test_acc - train_acc:.2%}")
        col_c.markdown(f"<div style='text-align:center; padding:0.8rem; border-radius:8px; background:{status_color}22; color:{status_color}; font-weight:bold; margin-top:0.3rem'>{status}</div>", unsafe_allow_html=True)

        # Depth vs Accuracy curve
        st.markdown("#### 📈 Accuracy vs Profundidad del árbol")
        depths = range(1, 16)
        train_accs, test_accs = [], []
        for d in depths:
            m = DecisionTreeClassifier(max_depth=d, random_state=42)
            m.fit(X_train, y_train)
            train_accs.append(accuracy_score(y_train, m.predict(X_train)))
            test_accs.append(accuracy_score(y_test, m.predict(X_test)))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=list(depths), y=train_accs, mode='lines+markers',
                                   name='Train', line=dict(color='#e74c3c', width=2)))
        fig2.add_trace(go.Scatter(x=list(depths), y=test_accs, mode='lines+markers',
                                   name='Test', line=dict(color='#2ecc71', width=2)))
        fig2.add_vline(x=max_depth, line_dash="dash", line_color="orange",
                       annotation_text=f"Depth={max_depth}")
        fig2.update_layout(xaxis_title="Profundidad", yaxis_title="Accuracy",
                            height=300, margin=dict(t=20, b=40))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### 📉 Curvas de Aprendizaje")
        st.markdown("""
        <div class="concept-box">
        Las <b>curvas de aprendizaje</b> muestran cómo cambia el error a medida que aumenta el tamaño del conjunto de entrenamiento.
        Son la herramienta diagnóstica más poderosa para detectar underfitting/overfitting.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            lc_depth = st.slider("Profundidad del árbol", 1, 15, 5, key="lc_depth")
            lc_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris (multiclase)"], key="lc_ds")

        if lc_dataset == "Breast Cancer":
            data = load_breast_cancer()
        else:
            data = load_iris()
        X_lc, y_lc = data.data, data.target

        model_lc = DecisionTreeClassifier(max_depth=lc_depth, random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            model_lc, X_lc, y_lc, cv=5,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', n_jobs=-1
        )

        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(train_sizes, train_scores.mean(axis=1), 'r-o', label='Entrenamiento', linewidth=2)
            ax.fill_between(train_sizes,
                             train_scores.mean(axis=1) - train_scores.std(axis=1),
                             train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.15, color='red')
            ax.plot(train_sizes, val_scores.mean(axis=1), 'g-o', label='Validación', linewidth=2)
            ax.fill_between(train_sizes,
                             val_scores.mean(axis=1) - val_scores.std(axis=1),
                             val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.15, color='green')
            ax.set_xlabel("Tamaño del conjunto de entrenamiento", fontsize=11)
            ax.set_ylabel("Accuracy", fontsize=11)
            ax.set_title(f"Curvas de Aprendizaje — Depth={lc_depth}", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

        train_final = train_scores.mean(axis=1)[-1]
        val_final = val_scores.mean(axis=1)[-1]
        gap = train_final - val_final

        if gap > 0.1:
            st.markdown(f"""<div class="warning-box">
            ⚠️ <b>Diagnóstico: OVERFITTING</b><br>
            Gap entrenamiento-validación = {gap:.2%}. Considera reducir la profundidad o usar regularización.
            </div>""", unsafe_allow_html=True)
        elif val_final < 0.75:
            st.markdown(f"""<div class="warning-box">
            🔵 <b>Diagnóstico: UNDERFITTING</b><br>
            Accuracy de validación = {val_final:.2%}. Considera aumentar la complejidad o agregar features.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="success-box">
            ✅ <b>Diagnóstico: Buen ajuste</b><br>
            Accuracy de validación = {val_final:.2%}, gap = {gap:.2%}. El modelo generaliza bien.
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MODULE 2: DECISION TREES
# ═══════════════════════════════════════════════════════

def show_decision_trees():
    st.markdown('<div class="section-title">🌲 Árboles de Decisión</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Estructura & Conceptos", "🎮 Constructor Interactivo", "🔍 Interpretación del Árbol"])

    with tab1:
        st.markdown("""
        <div class="concept-box">
        Un <b>árbol de decisión</b> es un modelo que toma decisiones mediante una serie de preguntas anidadas sobre los atributos de entrada.
        Cada nodo interno aplica una condición sobre una feature; cada rama representa el resultado de esa condición;
        cada hoja asigna una clase o valor.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🧩 Componentes del árbol")
            components = {
                "Nodo raíz (Root)": "El primer split; la pregunta más informativa",
                "Nodo interno (Split)": "Decisión intermedia sobre una feature",
                "Hoja (Leaf)": "Nodo terminal; contiene la predicción",
                "Profundidad (Depth)": "Número de niveles desde la raíz hasta la hoja más lejana",
                "Impureza": "Medida de mezcla de clases en un nodo (Gini, Entropía)",
            }
            for k, v in components.items():
                st.markdown(f"- **{k}**: {v}")

        with col2:
            st.markdown("#### ⚙️ Hiperparámetros clave")
            params = {
                "max_depth": "Profundidad máxima → controla overfitting",
                "min_samples_split": "Mínimo de muestras para hacer un split",
                "min_samples_leaf": "Mínimo de muestras en una hoja",
                "max_features": "Número de features a considerar en cada split",
                "criterion": "Función de impureza: 'gini' o 'entropy'",
                "ccp_alpha": "Parámetro de poda (cost-complexity pruning)",
            }
            for k, v in params.items():
                st.markdown(f"- `{k}`: {v}")

        st.markdown("""
        <div class="formula-box">
        <b>Algoritmo CART (Classification And Regression Trees):</b><br>
        1. Para cada feature y cada umbral posible, calcular la impureza del split<br>
        2. Elegir el split que más reduce la impureza (Information Gain / Gini Gain)<br>
        3. Repetir recursivamente en cada subconjunto<br>
        4. Detener cuando se cumpla una condición de parada (max_depth, min_samples, etc.)
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🎮 Construye y visualiza tu árbol")
        col1, col2 = st.columns([1, 2])

        with col1:
            dt_dataset = st.selectbox("Dataset", ["Iris", "Breast Cancer", "Moons sintético"], key="dt_ds")
            dt_criterion = st.selectbox("Criterio de impureza", ["gini", "entropy"], key="dt_crit")
            dt_max_depth = st.slider("max_depth", 1, 10, 3, key="dt_depth")
            dt_min_samples_split = st.slider("min_samples_split", 2, 50, 2, key="dt_mss")
            dt_min_samples_leaf = st.slider("min_samples_leaf", 1, 30, 1, key="dt_msl")
            dt_test_size = st.slider("Test size", 0.1, 0.5, 0.3, key="dt_ts")

        if dt_dataset == "Iris":
            data = load_iris()
            X, y = data.data[:, :2], data.target
            class_names = data.target_names
            feature_names = data.feature_names[:2]
        elif dt_dataset == "Breast Cancer":
            data = load_breast_cancer()
            X, y = data.data[:, :2], data.target
            class_names = data.target_names
            feature_names = data.feature_names[:2]
        else:
            X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
            class_names = ["Clase 0", "Clase 1"]
            feature_names = ["Feature 1", "Feature 2"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=dt_test_size, random_state=42, stratify=y)
        dt = DecisionTreeClassifier(
            criterion=dt_criterion, max_depth=dt_max_depth,
            min_samples_split=dt_min_samples_split, min_samples_leaf=dt_min_samples_leaf,
            random_state=42
        )
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)

        with col2:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            plot_decision_boundary(dt, X_train, y_train, "Frontera (Train)", axes[0])
            plot_decision_boundary(dt, X_test, y_test, f"Frontera (Test) Acc={accuracy_score(y_test,y_pred):.2%}", axes[1])
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("#### 🌳 Estructura visual del árbol")
        fig_tree, ax_tree = plt.subplots(figsize=(min(20, dt_max_depth * 4), min(10, dt_max_depth * 2.5)))
        plot_tree(dt, feature_names=feature_names, class_names=[str(c) for c in class_names],
                  filled=True, rounded=True, fontsize=8, ax=ax_tree, impurity=True, proportion=False)
        ax_tree.set_title(f"Árbol de Decisión — criterion={dt_criterion}, max_depth={dt_max_depth}", fontsize=12)
        st.pyplot(fig_tree)
        plt.close()

        st.markdown("#### 📊 Métricas rápidas")
        colm = st.columns(4)
        colm[0].metric("Accuracy (Test)", f"{accuracy_score(y_test, y_pred):.3f}")
        colm[1].metric("Nodos del árbol", dt.tree_.node_count)
        colm[2].metric("Profundidad real", dt.get_depth())
        colm[3].metric("Hojas", dt.get_n_leaves())

    with tab3:
        st.markdown("### 🔍 Importancia de Features & Reglas del árbol")
        feat_data = load_breast_cancer()
        X_f, y_f = feat_data.data, feat_data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X_f, y_f, test_size=0.3, random_state=42, stratify=y_f)

        fi_depth = st.slider("Profundidad para análisis de features", 1, 10, 4, key="fi_depth")
        dt_fi = DecisionTreeClassifier(max_depth=fi_depth, random_state=42)
        dt_fi.fit(X_tr, y_tr)

        importances = dt_fi.feature_importances_
        feat_names = feat_data.feature_names
        sorted_idx = np.argsort(importances)[::-1][:15]

        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        colors = ['#1a6b8a' if i == sorted_idx[0] else '#5ba3c9' for i in sorted_idx]
        ax_fi.barh(range(len(sorted_idx)), importances[sorted_idx], color=colors)
        ax_fi.set_yticks(range(len(sorted_idx)))
        ax_fi.set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=9)
        ax_fi.set_xlabel("Importancia (Gini Impurity Reduction)")
        ax_fi.set_title("Feature Importance — Breast Cancer Dataset", fontsize=12, fontweight='bold')
        ax_fi.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

        st.markdown("#### 📜 Reglas del árbol (primeras 30 líneas)")
        rules = export_text(dt_fi, feature_names=list(feat_names), max_depth=3)
        st.code(rules[:2000] + ("\n..." if len(rules) > 2000 else ""), language="text")


# ═══════════════════════════════════════════════════════
#  MODULE 3: ENTROPY & INFORMATION GAIN
# ═══════════════════════════════════════════════════════

def show_entropy():
    st.markdown('<div class="section-title">📐 Entropía & Information Gain</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🧮 Calculadora Interactiva", "🎮 Comparación Gini vs Entropía"])

    with tab1:
        st.markdown("""
        <div class="concept-box">
        Para elegir el <b>mejor split</b>, los árboles miden la <b>impureza</b> de los nodos.
        La pregunta que más reduce la impureza (mayor Information Gain) es la elegida.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="formula-box">
            <b>Entropía de Shannon:</b><br>
            H(π) = -Σ π·log₂(π)<br><br>
            Para clasificación binaria (p positivos, n negativos):<br>
            πₚ = p/(p+n) , πₙ = n/(p+n)<br>
            H(πₚ, πₙ) = -πₚ·log₂(πₚ) - πₙ·log₂(πₙ)<br><br>
            <b>Valores extremos:</b><br>
            • H=0 → nodo puro (una sola clase)<br>
            • H=1 → máxima impureza (50%/50%)
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="formula-box">
            <b>Índice Gini:</b><br>
            Gini = 1 - Σ πᵢ²<br><br>
            Para clasificación binaria:<br>
            Gini(p) = 1 - p² - (1-p)²  = 2p(1-p)<br><br>
            <b>Expected Entropy / Weighted Impurity:</b><br>
            EH(A) = Σᵢ [(pᵢ+nᵢ)/(p+n)] · H(πₚᵢ, πₙᵢ)<br><br>
            <b>Information Gain:</b><br>
            IG(A) = H(πₚ, πₙ) - EH(A)
            </div>
            """, unsafe_allow_html=True)

        # Plot both curves
        p_vals = np.linspace(0.001, 0.999, 200)
        ent_vals = [entropy(p) for p in p_vals]
        gini_vals = [gini(p) for p in p_vals]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(p_vals, ent_vals, 'b-', linewidth=2.5, label='Entropía H(p)')
        ax.plot(p_vals, gini_vals, 'r-', linewidth=2.5, label='Gini 2p(1-p)')
        ax.set_xlabel("Proporción de clase positiva (p)", fontsize=12)
        ax.set_ylabel("Impureza", fontsize=12)
        ax.set_title("Entropía vs Gini: Medidas de Impureza", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.annotate('Máxima impureza\n(p=0.5)', xy=(0.5, 1.0), xytext=(0.6, 0.8),
                     arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10, color='gray')
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("### 🧮 Calcula el Information Gain manualmente")
        st.markdown("Simula un nodo padre y dos nodos hijo para entender cómo se elige el mejor split.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Nodo Padre**")
            p_pos = st.number_input("Positivos (padre)", 1, 100, 6, key="ep")
            p_neg = st.number_input("Negativos (padre)", 1, 100, 6, key="en")

        with col2:
            st.markdown("**Hijo Izquierdo**")
            l_pos = st.number_input("Positivos (izq)", 0, 100, 4, key="lp")
            l_neg = st.number_input("Negativos (izq)", 0, 100, 0, key="ln")

        with col3:
            st.markdown("**Hijo Derecho**")
            r_pos = st.number_input("Positivos (der)", 0, 100, 2, key="rp")
            r_neg = st.number_input("Negativos (der)", 0, 100, 6, key="rn")

        total = p_pos + p_neg
        h_parent = entropy(p_pos / total) if total > 0 else 0
        total_l = l_pos + l_neg
        total_r = r_pos + r_neg
        h_left = entropy(l_pos / total_l) if total_l > 0 else 0
        h_right = entropy(r_pos / total_r) if total_r > 0 else 0
        eh = (total_l / total) * h_left + (total_r / total) * h_right if total > 0 else 0
        ig = h_parent - eh

        gini_parent = gini(p_pos / total) if total > 0 else 0
        gini_left = gini(l_pos / total_l) if total_l > 0 else 0
        gini_right = gini(r_pos / total_r) if total_r > 0 else 0
        eg = (total_l / total) * gini_left + (total_r / total) * gini_right if total > 0 else 0
        gini_gain = gini_parent - eg

        st.markdown("---")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        col_r1.metric("H(Padre)", f"{h_parent:.4f}")
        col_r2.metric("EH(Split)", f"{eh:.4f}")
        col_r3.metric("Information Gain (IG)", f"{ig:.4f}", delta="↑ mejor si mayor")
        col_r4.metric("Gini Gain", f"{gini_gain:.4f}", delta="↑ mejor si mayor")

        fig_bar, ax_bar = plt.subplots(figsize=(7, 3))
        labels = ['H(Padre)', 'H(Izq)', 'H(Der)', 'EH(Split)']
        values = [h_parent, h_left, h_right, eh]
        colors_bar = ['#1a6b8a', '#5ba3c9', '#5ba3c9', '#e74c3c']
        ax_bar.bar(labels, values, color=colors_bar, edgecolor='white')
        ax_bar.set_ylabel("Entropía")
        ax_bar.set_title(f"Information Gain = {ig:.4f}", fontsize=12, fontweight='bold')
        ax_bar.grid(axis='y', alpha=0.3)
        st.pyplot(fig_bar)
        plt.close()

    with tab3:
        st.markdown("### 🎮 ¿Gini o Entropía? ¿Cambia el árbol?")
        gc_dataset = st.selectbox("Dataset", ["Iris", "Breast Cancer"], key="gc_ds")
        gc_depth = st.slider("max_depth", 1, 8, 3, key="gc_d")

        if gc_dataset == "Iris":
            data = load_iris(); fn = list(data.feature_names[:2]); cn = list(data.target_names)
            X_gc, y_gc = data.data[:, :2], data.target
        else:
            data = load_breast_cancer(); fn = list(data.feature_names[:2]); cn = list(data.target_names)
            X_gc, y_gc = data.data[:, :2], data.target

        X_tr, X_te, y_tr, y_te = train_test_split(X_gc, y_gc, test_size=0.3, random_state=42, stratify=y_gc)
        dt_gini = DecisionTreeClassifier(criterion='gini', max_depth=gc_depth, random_state=42).fit(X_tr, y_tr)
        dt_ent = DecisionTreeClassifier(criterion='entropy', max_depth=gc_depth, random_state=42).fit(X_tr, y_tr)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(dt_gini, X_te, y_te, f"Gini — Acc={accuracy_score(y_te, dt_gini.predict(X_te)):.2%}", axes[0])
        plot_decision_boundary(dt_ent, X_te, y_te, f"Entropy — Acc={accuracy_score(y_te, dt_ent.predict(X_te)):.2%}", axes[1])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="concept-box">
        💡 <b>Conclusión práctica:</b> En la mayoría de los casos, Gini y Entropía producen resultados muy similares.
        Gini es ligeramente más rápido computacionalmente (no requiere logaritmos). Entropía puede producir
        árboles más balanceados en algunos datasets.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MODULE 4: METRICS
# ═══════════════════════════════════════════════════════

def show_metrics():
    st.markdown('<div class="section-title">📏 Métricas de Desempeño</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Matriz de Confusión", "📈 ROC & PR Curves", "🧮 Calculadora de Métricas"])

    with tab1:
        st.markdown("""
        <div class="concept-box">
        La <b>matriz de confusión</b> es la base de todas las métricas de clasificación.
        Muestra cuántas muestras fueron clasificadas correctamente o incorrectamente para cada clase.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div class="formula-box">
            <b>Métricas derivadas (problema binario):</b><br><br>
            • <b>Accuracy</b> = (TP + TN) / Total<br>
            • <b>Precision</b> = TP / (TP + FP)  → ¿Cuántos de los predichos positivos son realmente positivos?<br>
            • <b>Recall / Sensitivity</b> = TP / (TP + FN) → ¿Cuántos positivos reales detecté?<br>
            • <b>Specificity</b> = TN / (TN + FP)<br>
            • <b>F1-Score</b> = 2 · (Precision · Recall) / (Precision + Recall)<br>
            • <b>ROC-AUC</b> = Área bajo la curva ROC
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>¿Cuándo NO usar Accuracy?</b><br><br>
            Con clases desbalanceadas, un modelo que siempre predice la clase mayoritaria puede tener
            Accuracy muy alta, pero ser completamente inútil.<br><br>
            Ejemplo: 95% clase 0, 5% clase 1 → predecir siempre 0 da Accuracy=95% pero Recall=0% para clase 1.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎮 Genera matriz de confusión interactiva")
        cm_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris"], key="cm_ds")
        cm_depth = st.slider("Profundidad del árbol", 1, 15, 4, key="cm_d")

        if cm_dataset == "Breast Cancer":
            data = load_breast_cancer(); cn = data.target_names
            X_cm, y_cm = data.data, data.target
        else:
            data = load_iris(); cn = data.target_names
            X_cm, y_cm = data.data, data.target

        X_tr, X_te, y_tr, y_te = train_test_split(X_cm, y_cm, test_size=0.3, random_state=42, stratify=y_cm)
        model_cm = DecisionTreeClassifier(max_depth=cm_depth, random_state=42).fit(X_tr, y_tr)
        y_pred_cm = model_cm.predict(X_te)
        cm = confusion_matrix(y_te, y_pred_cm)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=cn, yticklabels=cn, linewidths=0.5)
        ax_cm.set_xlabel("Predicho", fontsize=11)
        ax_cm.set_ylabel("Real", fontsize=11)
        ax_cm.set_title("Matriz de Confusión", fontsize=13, fontweight='bold')
        st.pyplot(fig_cm)
        plt.close()

        if len(np.unique(y_cm)) == 2:
            col_m = st.columns(5)
            col_m[0].metric("Accuracy", f"{accuracy_score(y_te, y_pred_cm):.3f}")
            col_m[1].metric("Precision", f"{precision_score(y_te, y_pred_cm):.3f}")
            col_m[2].metric("Recall", f"{recall_score(y_te, y_pred_cm):.3f}")
            col_m[3].metric("F1-Score", f"{f1_score(y_te, y_pred_cm):.3f}")
            col_m[4].metric("ROC-AUC", f"{roc_auc_score(y_te, model_cm.predict_proba(X_te)[:, 1]):.3f}")

        st.markdown("#### 📋 Reporte completo")
        report = classification_report(y_te, y_pred_cm, target_names=[str(c) for c in cn])
        st.code(report, language="text")

    with tab2:
        st.markdown("### 📈 Curvas ROC y Precision-Recall")
        st.markdown("""
        <div class="concept-box">
        <b>Curva ROC</b>: grafica TPR (Recall) vs FPR para distintos umbrales de clasificación.<br>
        <b>Curva PR</b>: grafica Precision vs Recall. Más informativa cuando hay clases desbalanceadas.<br>
        <b>AUC</b>: Área bajo la curva. AUC=1 es perfecto, AUC=0.5 es aleatorio.
        </div>
        """, unsafe_allow_html=True)

        data = load_breast_cancer()
        X_roc, y_roc = data.data, data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X_roc, y_roc, test_size=0.3, random_state=42, stratify=y_roc)

        depths_roc = [2, 4, 6, None]
        labels_roc = ['depth=2', 'depth=4', 'depth=6', 'depth=None (full)']
        colors_roc = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

        fig_roc = make_subplots(rows=1, cols=2, subplot_titles=["Curva ROC", "Curva Precision-Recall"])

        for d, label, color in zip(depths_roc, labels_roc, colors_roc):
            m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = roc_auc_score(y_te, proba)
            prec, rec, _ = precision_recall_curve(y_te, proba)
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{label} (AUC={auc:.2f})",
                                          line=dict(color=color, width=2)), row=1, col=1)
            fig_roc.add_trace(go.Scatter(x=rec, y=prec, name=label, showlegend=False,
                                          line=dict(color=color, width=2, dash='dot')), row=1, col=2)

        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random', line=dict(dash='dash', color='gray')), row=1, col=1)
        fig_roc.update_xaxes(title_text="FPR (1 - Specificity)", row=1, col=1)
        fig_roc.update_yaxes(title_text="TPR (Recall)", row=1, col=1)
        fig_roc.update_xaxes(title_text="Recall", row=1, col=2)
        fig_roc.update_yaxes(title_text="Precision", row=1, col=2)
        fig_roc.update_layout(height=420, margin=dict(t=40, b=40))
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.markdown("### 🧮 Calculadora de Métricas desde Matriz de Confusión")
        col1, col2 = st.columns(2)
        with col1:
            tp = st.number_input("TP (True Positives)", 0, 1000, 85)
            fp = st.number_input("FP (False Positives)", 0, 1000, 10)
        with col2:
            fn = st.number_input("FN (False Negatives)", 0, 1000, 15)
            tn = st.number_input("TN (True Negatives)", 0, 1000, 90)

        total = tp + fp + fn + tn
        if total > 0:
            acc = (tp + tn) / total
            prec_calc = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec_calc = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1_calc = 2 * prec_calc * rec_calc / (prec_calc + rec_calc) if (prec_calc + rec_calc) > 0 else 0
            fpr_calc = fp / (fp + tn) if (fp + tn) > 0 else 0

            cols_calc = st.columns(3)
            cols_calc[0].metric("Accuracy", f"{acc:.3f}")
            cols_calc[1].metric("Precision", f"{prec_calc:.3f}")
            cols_calc[2].metric("Recall / Sensitivity", f"{rec_calc:.3f}")
            cols_calc2 = st.columns(3)
            cols_calc2[0].metric("Specificity", f"{spec:.3f}")
            cols_calc2[1].metric("F1-Score", f"{f1_calc:.3f}")
            cols_calc2[2].metric("FPR (1-Specificity)", f"{fpr_calc:.3f}")

            fig_cm2, ax_cm2 = plt.subplots(figsize=(4, 3))
            cm_manual = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Blues', ax=ax_cm2,
                        xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['Real Neg', 'Real Pos'])
            ax_cm2.set_title("Tu Matriz de Confusión")
            st.pyplot(fig_cm2)
            plt.close()


# ═══════════════════════════════════════════════════════
#  MODULE 5: VALIDATION & CROSS-VALIDATION
# ═══════════════════════════════════════════════════════

def show_validation():
    st.markdown('<div class="section-title">✅ Validación & Cross-Validation</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Estrategias de Validación", "🎮 K-Fold Interactivo", "🔍 Validation Curve"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-box">
            <b>1. Hold-Out (Train/Test Split)</b><br>
            Divide los datos en un conjunto de entrenamiento y uno de prueba (ej. 70/30).<br>
            • ✅ Rápido y simple<br>
            • ❌ Alta varianza (depende de cómo se hizo la división)<br>
            • ❌ No aprovecha todos los datos para entrenamiento
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="concept-box">
            <b>2. K-Fold Cross-Validation</b><br>
            Divide en K partes iguales (folds). Entrena K veces usando K-1 partes y evalúa en la restante.<br>
            • ✅ Estimación más robusta del error<br>
            • ✅ Usa todos los datos para entrenar y validar<br>
            • ❌ K veces más costoso computacionalmente
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="concept-box">
            <b>3. Stratified K-Fold</b><br>
            Igual que K-Fold pero garantiza que cada fold tenga la misma proporción de clases.<br>
            • ✅ Esencial para datasets desbalanceados<br>
            • ✅ Estadísticamente más correcto para clasificación
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="concept-box">
            <b>4. Leave-One-Out (LOO)</b><br>
            Caso extremo de K-Fold donde K = N (número de muestras).<br>
            • ✅ Máximo uso de datos<br>
            • ❌ Muy costoso para datasets grandes<br>
            • Útil solo para datasets muy pequeños
            </div>
            """, unsafe_allow_html=True)

        # Diagram of K-Fold
        st.markdown("#### 🖼️ Diagrama K-Fold (K=5)")
        fig_kf, ax_kf = plt.subplots(figsize=(10, 3))
        K = 5
        for fold in range(K):
            for part in range(K):
                color = '#e74c3c' if part == fold else '#2ecc71'
                label = 'Test' if part == fold else 'Train'
                rect = mpatches.Rectangle([part / K, (K - fold - 1) / K + 0.05],
                                           1 / K - 0.01, 0.8 / K, color=color, alpha=0.7)
                ax_kf.add_patch(rect)
                ax_kf.text(part / K + 0.5 / K, (K - fold - 1) / K + 0.05 + 0.4 / K,
                            label[:2], ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        for fold in range(K):
            ax_kf.text(-0.02, (K - fold - 1) / K + 0.05 + 0.4 / K, f'Fold {fold+1}', ha='right', va='center', fontsize=9)
        ax_kf.set_xlim(-0.1, 1.05); ax_kf.set_ylim(0, 1.1)
        ax_kf.axis('off')
        ax_kf.set_title("K-Fold Cross-Validation (K=5)", fontsize=12, fontweight='bold')
        train_patch = mpatches.Patch(color='#2ecc71', alpha=0.7, label='Entrenamiento')
        test_patch = mpatches.Patch(color='#e74c3c', alpha=0.7, label='Validación')
        ax_kf.legend(handles=[train_patch, test_patch], loc='lower right')
        st.pyplot(fig_kf)
        plt.close()

    with tab2:
        st.markdown("### 🎮 Observa la variabilidad entre folds")
        cv_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris"], key="cv_ds")
        cv_depth = st.slider("max_depth", 1, 15, 4, key="cv_depth")
        k_folds = st.slider("Número de folds (K)", 3, 20, 5, key="cv_k")

        if cv_dataset == "Breast Cancer":
            data = load_breast_cancer()
        else:
            data = load_iris()
        X_cv, y_cv = data.data, data.target

        model_cv = DecisionTreeClassifier(max_depth=cv_depth, random_state=42)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model_cv, X_cv, y_cv, cv=kf, scoring='accuracy')

        fig_cv, ax_cv = plt.subplots(figsize=(9, 4))
        fold_nums = list(range(1, k_folds + 1))
        colors_cv = ['#e74c3c' if s < scores.mean() - scores.std() else '#2ecc71' if s > scores.mean() + scores.std() else '#3498db' for s in scores]
        bars = ax_cv.bar(fold_nums, scores, color=colors_cv, edgecolor='white', linewidth=1.5)
        ax_cv.axhline(scores.mean(), color='black', linestyle='--', linewidth=2, label=f'Media = {scores.mean():.3f}')
        ax_cv.axhspan(scores.mean() - scores.std(), scores.mean() + scores.std(), alpha=0.15, color='gray', label=f'±1 std = {scores.std():.3f}')
        ax_cv.set_xlabel("Fold", fontsize=11)
        ax_cv.set_ylabel("Accuracy", fontsize=11)
        ax_cv.set_title(f"Cross-Validation Scores — {k_folds}-Fold", fontsize=12, fontweight='bold')
        ax_cv.set_xticks(fold_nums)
        ax_cv.legend(fontsize=10)
        ax_cv.set_ylim(max(0, scores.min() - 0.1), min(1.05, scores.max() + 0.05))
        ax_cv.grid(axis='y', alpha=0.3)
        for bar, score in zip(bars, scores):
            ax_cv.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        st.pyplot(fig_cv)
        plt.close()

        col_cv = st.columns(4)
        col_cv[0].metric("Media", f"{scores.mean():.4f}")
        col_cv[1].metric("Desviación Estándar", f"{scores.std():.4f}")
        col_cv[2].metric("Mínimo", f"{scores.min():.4f}")
        col_cv[3].metric("Máximo", f"{scores.max():.4f}")

        st.markdown(f"""
        <div class="concept-box">
        📊 <b>Resultado de la validación:</b> El modelo tiene una accuracy promedio de <b>{scores.mean():.2%} ± {scores.std():.2%}</b>.
        La desviación estándar indica la <b>estabilidad</b> del modelo: valores bajos = modelo robusto.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🔍 Validation Curve: encontrando el hiperparámetro óptimo")
        st.markdown("""
        <div class="concept-box">
        La <b>Validation Curve</b> muestra cómo cambia el desempeño de entrenamiento y validación
        a medida que varía un hiperparámetro. Es la herramienta para encontrar el valor óptimo.
        </div>
        """, unsafe_allow_html=True)

        vc_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris"], key="vc_ds")
        vc_param = st.selectbox("Hiperparámetro", ["max_depth", "min_samples_split", "min_samples_leaf"], key="vc_p")

        if vc_dataset == "Breast Cancer":
            data = load_breast_cancer()
        else:
            data = load_iris()
        X_vc, y_vc = data.data, data.target

        if vc_param == "max_depth":
            param_range = np.arange(1, 16)
        elif vc_param == "min_samples_split":
            param_range = np.arange(2, 50, 3)
        else:
            param_range = np.arange(1, 30, 2)

        train_sc, val_sc = validation_curve(
            DecisionTreeClassifier(random_state=42), X_vc, y_vc,
            param_name=vc_param, param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )

        fig_vc, ax_vc = plt.subplots(figsize=(9, 4))
        ax_vc.plot(param_range, train_sc.mean(axis=1), 'r-o', linewidth=2, markersize=5, label='Entrenamiento')
        ax_vc.fill_between(param_range,
                            train_sc.mean(axis=1) - train_sc.std(axis=1),
                            train_sc.mean(axis=1) + train_sc.std(axis=1), alpha=0.15, color='red')
        ax_vc.plot(param_range, val_sc.mean(axis=1), 'g-o', linewidth=2, markersize=5, label='Validación (CV)')
        ax_vc.fill_between(param_range,
                            val_sc.mean(axis=1) - val_sc.std(axis=1),
                            val_sc.mean(axis=1) + val_sc.std(axis=1), alpha=0.15, color='green')
        best_idx = np.argmax(val_sc.mean(axis=1))
        ax_vc.axvline(param_range[best_idx], color='orange', linestyle='--',
                       label=f'Mejor: {vc_param}={param_range[best_idx]}')
        ax_vc.set_xlabel(vc_param, fontsize=11)
        ax_vc.set_ylabel("Accuracy", fontsize=11)
        ax_vc.set_title(f"Validation Curve — {vc_param}", fontsize=12, fontweight='bold')
        ax_vc.legend(fontsize=10)
        ax_vc.grid(alpha=0.3)
        st.pyplot(fig_vc)
        plt.close()

        st.markdown(f"""
        <div class="success-box">
        ✅ <b>Mejor valor encontrado:</b> <code>{vc_param} = {param_range[best_idx]}</code>
        con Accuracy de validación = <b>{val_sc.mean(axis=1)[best_idx]:.4f}</b>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MODULE 6: IMBALANCED CLASSES
# ═══════════════════════════════════════════════════════

def show_imbalanced():
    st.markdown('<div class="section-title">⚖️ Clases Desbalanceadas</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📖 Técnicas de Balanceo", "🎮 Comparación Interactiva"])

    with tab1:
        st.markdown("""
        <div class="concept-box">
        En problemas reales (fraude, diagnóstico médico, spam), la clase de interés puede representar
        solo el 1-5% de los datos. Un clasificador naïve que siempre predice la clase mayoritaria
        tendrá alta Accuracy pero <b>Recall = 0 para la clase minoritaria</b>.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="concept-box">
            <b>1. Oversampling</b><br>
            Aumenta la clase minoritaria.<br><br>
            • <b>Random Oversampling</b>: duplica muestras existentes<br>
            • <b>SMOTE</b>: crea muestras sintéticas interpolando entre vecinos cercanos<br>
            • <b>ADASYN</b>: SMOTE adaptativo, más muestras donde hay más dificultad
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-box">
            <b>2. Undersampling</b><br>
            Reduce la clase mayoritaria.<br><br>
            • <b>Random Undersampling</b>: elimina muestras aleatoriamente<br>
            • <b>Tomek Links</b>: elimina pares de clases opuestas muy cercanos<br>
            • <b>NearMiss</b>: selecciona muestras mayoritarias cercanas a la frontera
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="concept-box">
            <b>3. Pesos de Clase</b><br>
            Penaliza más los errores en la clase minoritaria.<br><br>
            • En sklearn: <code>class_weight='balanced'</code><br>
            • Equivale matemáticamente a oversampling<br>
            • ✅ No cambia el dataset, solo el entrenamiento
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🎮 Experimenta con técnicas de balanceo")
        col1, col2 = st.columns([1, 2])
        with col1:
            imb_ratio = st.slider("Ratio de desbalanceo (% clase minoritaria)", 2, 50, 10)
            n_imb = st.slider("Número de muestras totales", 200, 2000, 500, 100)
            technique = st.selectbox("Técnica de balanceo",
                                      ["Sin balanceo", "SMOTE", "Undersampling", "class_weight='balanced'"])
            imb_depth = st.slider("max_depth del árbol", 1, 15, 5, key="imb_d")

        n_class0 = int(n_imb * (1 - imb_ratio / 100))
        n_class1 = int(n_imb * imb_ratio / 100)
        X_imb = np.vstack([np.random.randn(n_class0, 2) + [2, 2],
                            np.random.randn(n_class1, 2) + [0, 0]])
        y_imb = np.hstack([np.zeros(n_class0), np.ones(n_class1)])
        idx = np.random.permutation(len(y_imb))
        X_imb, y_imb = X_imb[idx], y_imb[idx]

        X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42, stratify=y_imb)

        if technique == "SMOTE":
            try:
                sm = SMOTE(random_state=42)
                X_tr_bal, y_tr_bal = sm.fit_resample(X_tr_i, y_tr_i)
                cw = None
            except Exception:
                X_tr_bal, y_tr_bal = X_tr_i, y_tr_i; cw = None
        elif technique == "Undersampling":
            rus = RandomUnderSampler(random_state=42)
            X_tr_bal, y_tr_bal = rus.fit_resample(X_tr_i, y_tr_i)
            cw = None
        elif technique == "class_weight='balanced'":
            X_tr_bal, y_tr_bal = X_tr_i, y_tr_i
            cw = 'balanced'
        else:
            X_tr_bal, y_tr_bal = X_tr_i, y_tr_i
            cw = None

        model_imb = DecisionTreeClassifier(max_depth=imb_depth, random_state=42, class_weight=cw).fit(X_tr_bal, y_tr_bal)
        y_pred_i = model_imb.predict(X_te_i)

        with col2:
            fig_imb, axes_imb = plt.subplots(1, 2, figsize=(12, 5))
            val_counts = pd.Series(y_tr_bal).value_counts().sort_index()
            axes_imb[0].bar(['Clase 0', 'Clase 1'], val_counts.values,
                             color=['#3498db', '#e74c3c'], edgecolor='white')
            axes_imb[0].set_title(f"Distribución en Train\n({technique})", fontsize=11, fontweight='bold')
            axes_imb[0].set_ylabel("Muestras")
            for i, v in enumerate(val_counts.values):
                axes_imb[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

            cm_imb = confusion_matrix(y_te_i, y_pred_i)
            sns.heatmap(cm_imb, annot=True, fmt='d', cmap='Blues', ax=axes_imb[1],
                        xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Real 0', 'Real 1'])
            axes_imb[1].set_title("Matriz de Confusión (Test)", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_imb)
            plt.close()

        rec_cls1 = recall_score(y_te_i, y_pred_i, zero_division=0)
        prec_cls1 = precision_score(y_te_i, y_pred_i, zero_division=0)
        f1_cls1 = f1_score(y_te_i, y_pred_i, zero_division=0)
        acc_i = accuracy_score(y_te_i, y_pred_i)

        col_im = st.columns(4)
        col_im[0].metric("Accuracy", f"{acc_i:.3f}")
        col_im[1].metric("Precision (clase 1)", f"{prec_cls1:.3f}")
        col_im[2].metric("Recall (clase 1)", f"{rec_cls1:.3f}", delta="↑ crítico")
        col_im[3].metric("F1 (clase 1)", f"{f1_cls1:.3f}")


# ═══════════════════════════════════════════════════════
#  MODULE 7: BAGGING
# ═══════════════════════════════════════════════════════

def show_bagging():
    st.markdown('<div class="section-title">🌳🌳 Bootstrap Aggregating (Bagging) & Random Forest</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Teoría Bagging", "🌲 Random Forest Interactivo", "📊 Feature Importance"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-box">
            <b>Bagging (Bootstrap AGGregating)</b><br><br>
            1. Crea <b>B subconjuntos</b> de entrenamiento mediante bootstrapping (muestreo con reemplazo)<br>
            2. Entrena un modelo base en cada subconjunto <b>de forma independiente (paralela)</b><br>
            3. Combina predicciones:<br>
               • Regresión: promedio<br>
               • Clasificación: votación mayoritaria<br><br>
            <b>Objetivo:</b> reducir la varianza sin aumentar el sesgo
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-box">
            <b>Random Forest = Bagging + Feature Subsampling</b><br><br>
            Random Forest añade una capa extra de aleatoriedad:<br>
            • En cada split, solo considera un <b>subconjunto aleatorio de features</b><br>
            • Típicamente: <code>√n_features</code> para clasificación<br><br>
            Esto <b>descorelaciona</b> los árboles entre sí, reduciendo aún más la varianza.<br><br>
            Ventaja adicional: <b>Out-of-Bag (OOB) Score</b> — validación gratuita sin separar test set
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        <b>¿Por qué el promedio de modelos reduce varianza?</b><br>
        Si tenemos B modelos independientes con varianza σ²:<br>
        Var(promedio) = σ²/B → a más modelos, menor varianza<br><br>
        Pero los árboles no son independientes (usan el mismo dataset) → 
        correlación ρ entre árboles: Var(promedio) = ρσ² + (1-ρ)σ²/B
        Random Forest reduce ρ al aleatorizar las features.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🌲 Construye un Random Forest")
        col1, col2 = st.columns([1, 2])
        with col1:
            rf_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris", "Moons"], key="rf_ds")
            n_estimators = st.slider("Número de árboles", 1, 200, 50, key="rf_ne")
            rf_max_depth = st.slider("max_depth de cada árbol", 1, 20, 5, key="rf_md")
            max_features_opt = st.selectbox("max_features", ["sqrt", "log2", "None (todas)"], key="rf_mf")
            rf_bootstrap = st.checkbox("Bootstrap", value=True, key="rf_boot")

        mf = None if max_features_opt == "None (todas)" else max_features_opt

        if rf_dataset == "Breast Cancer":
            data = load_breast_cancer(); cn = data.target_names
            X_rf, y_rf = data.data, data.target
        elif rf_dataset == "Iris":
            data = load_iris(); cn = data.target_names
            X_rf, y_rf = data.data, data.target
        else:
            X_rf, y_rf = make_moons(n_samples=500, noise=0.25, random_state=42)
            cn = ["Clase 0", "Clase 1"]

        X_tr_rf, X_te_rf, y_tr_rf, y_te_rf = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf)

        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=rf_max_depth,
                                     max_features=mf, bootstrap=rf_bootstrap,
                                     oob_score=rf_bootstrap, random_state=42, n_jobs=-1)
        rf.fit(X_tr_rf, y_tr_rf)
        y_pred_rf = rf.predict(X_te_rf)

        with col2:
            # Accuracy vs n_estimators
            n_range = [1, 5, 10, 20, 30, 50, 75, 100, 150, 200]
            n_range = [n for n in n_range if n <= 200]
            accs_n = []
            for n in n_range:
                m = RandomForestClassifier(n_estimators=n, max_depth=rf_max_depth,
                                            max_features=mf, bootstrap=rf_bootstrap,
                                            random_state=42, n_jobs=-1).fit(X_tr_rf, y_tr_rf)
                accs_n.append(accuracy_score(y_te_rf, m.predict(X_te_rf)))

            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(x=n_range, y=accs_n, mode='lines+markers',
                                         name='Accuracy Test', line=dict(color='#2ecc71', width=2)))
            fig_rf.add_vline(x=n_estimators, line_dash="dash", line_color="orange",
                              annotation_text=f"n={n_estimators}")
            fig_rf.update_layout(xaxis_title="Número de árboles", yaxis_title="Accuracy",
                                  title="Accuracy vs Número de Árboles", height=350, margin=dict(t=40))
            st.plotly_chart(fig_rf, use_container_width=True)

        col_rf = st.columns(4 if rf_bootstrap else 3)
        col_rf[0].metric("Accuracy (Test)", f"{accuracy_score(y_te_rf, y_pred_rf):.4f}")
        col_rf[1].metric("F1 (macro)", f"{f1_score(y_te_rf, y_pred_rf, average='macro'):.4f}")
        col_rf[2].metric("Árboles", n_estimators)
        if rf_bootstrap:
            col_rf[3].metric("OOB Score", f"{rf.oob_score_:.4f}")

        if rf_dataset != "Moons":
            st.markdown("#### Comparación: Árbol solo vs Random Forest")
            single = DecisionTreeClassifier(max_depth=rf_max_depth, random_state=42).fit(X_tr_rf, y_tr_rf)
            single_acc = accuracy_score(y_te_rf, single.predict(X_te_rf))
            rf_acc = accuracy_score(y_te_rf, y_pred_rf)
            improvement = rf_acc - single_acc
            st.markdown(f"""
            <div class="{'success-box' if improvement > 0 else 'warning-box'}">
            🌲 Árbol individual: <b>{single_acc:.4f}</b> | 
            🌳🌳 Random Forest: <b>{rf_acc:.4f}</b> | 
            Mejora: <b>{improvement:+.4f}</b>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 📊 Feature Importance en Random Forest")
        fi_data = load_breast_cancer()
        X_fi, y_fi = fi_data.data, fi_data.target
        X_tr_fi, X_te_fi, y_tr_fi, y_te_fi = train_test_split(X_fi, y_fi, test_size=0.3, random_state=42, stratify=y_fi)

        n_trees_fi = st.slider("Número de árboles para feature importance", 10, 300, 100, key="fi_n")
        rf_fi = RandomForestClassifier(n_estimators=n_trees_fi, random_state=42, n_jobs=-1).fit(X_tr_fi, y_tr_fi)

        importances_rf = rf_fi.feature_importances_
        stds = np.std([tree.feature_importances_ for tree in rf_fi.estimators_], axis=0)
        sorted_idx_rf = np.argsort(importances_rf)[::-1][:15]

        fig_fi_rf, ax_fi_rf = plt.subplots(figsize=(9, 6))
        ax_fi_rf.barh(range(len(sorted_idx_rf)), importances_rf[sorted_idx_rf],
                       xerr=stds[sorted_idx_rf], color='#1a6b8a', alpha=0.8,
                       error_kw=dict(ecolor='black', capsize=3))
        ax_fi_rf.set_yticks(range(len(sorted_idx_rf)))
        ax_fi_rf.set_yticklabels([fi_data.feature_names[i] for i in sorted_idx_rf], fontsize=9)
        ax_fi_rf.set_xlabel("Importancia media ± std")
        ax_fi_rf.set_title(f"Feature Importance — Random Forest ({n_trees_fi} árboles)\nBreast Cancer Dataset",
                            fontsize=12, fontweight='bold')
        ax_fi_rf.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_fi_rf)
        plt.close()


# ═══════════════════════════════════════════════════════
#  MODULE 8: BOOSTING
# ═══════════════════════════════════════════════════════

def show_boosting():
    st.markdown('<div class="section-title">🚀 Boosting & Gradient Boosting</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Teoría Boosting", "🎮 Gradient Boosting Interactivo", "⚡ Comparación de Algoritmos"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="concept-box">
            <b>Boosting — Idea General</b><br><br>
            Modelos débiles entrenados <b>secuencialmente</b>:<br>
            1. Entrena modelo₁ con los datos originales<br>
            2. Identifica los errores de modelo₁<br>
            3. Entrena modelo₂ dando más peso a los ejemplos mal clasificados<br>
            4. Repite hasta B modelos<br>
            5. Combinación <b>ponderada</b> de todas las predicciones<br><br>
            <b>Objetivo:</b> reducir el sesgo (underfitting)
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="concept-box">
            <b>Gradient Boosting</b><br><br>
            En lugar de ponderar muestras, cada árbol predice el <b>residual</b> del ensamble anterior:<br><br>
            r₁ = y - ŷ₁<br>
            Modelo₂ predice r₁<br>
            r₂ = r₁ - r̂₁<br>
            Modelo₃ predice r₂ ...<br><br>
            Predicción final: F(x) = η·Σ fₜ(x)<br>
            η = <b>learning rate</b> (controla sobreajuste)
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="formula-box">
        <b>Diferencia Bagging vs Boosting:</b><br>
        • Bagging: modelos en <b>paralelo</b>, reduce varianza, usa bootstrapping<br>
        • Boosting: modelos en <b>secuencia</b>, reduce sesgo, pondera errores<br><br>
        <b>Variantes populares:</b> AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🎮 Explora Gradient Boosting")
        col1, col2 = st.columns([1, 2])
        with col1:
            gb_dataset = st.selectbox("Dataset", ["Breast Cancer", "Moons", "Iris"], key="gb_ds")
            gb_n = st.slider("n_estimators", 10, 300, 100, key="gb_n")
            gb_lr = st.select_slider("learning_rate", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], value=0.1, key="gb_lr")
            gb_depth = st.slider("max_depth (árboles base)", 1, 10, 3, key="gb_d")
            gb_subsample = st.slider("subsample", 0.3, 1.0, 1.0, 0.1, key="gb_sub")

        if gb_dataset == "Breast Cancer":
            data = load_breast_cancer(); cn = data.target_names
            X_gb, y_gb = data.data, data.target
        elif gb_dataset == "Moons":
            X_gb, y_gb = make_moons(n_samples=500, noise=0.25, random_state=42)
            cn = ["Clase 0", "Clase 1"]
        else:
            data = load_iris(); cn = data.target_names
            X_gb, y_gb = data.data, data.target

        X_tr_gb, X_te_gb, y_tr_gb, y_te_gb = train_test_split(X_gb, y_gb, test_size=0.3, random_state=42, stratify=y_gb)

        gb = GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr,
                                         max_depth=gb_depth, subsample=gb_subsample,
                                         random_state=42)
        gb.fit(X_tr_gb, y_tr_gb)

        staged_train = list(gb.staged_score(X_tr_gb, y_tr_gb))
        staged_test = list(gb.staged_score(X_te_gb, y_te_gb))

        with col2:
            fig_gb = go.Figure()
            fig_gb.add_trace(go.Scatter(y=staged_train, mode='lines', name='Train',
                                         line=dict(color='#e74c3c', width=2)))
            fig_gb.add_trace(go.Scatter(y=staged_test, mode='lines', name='Test',
                                         line=dict(color='#2ecc71', width=2)))
            best_iter = np.argmax(staged_test)
            fig_gb.add_vline(x=best_iter, line_dash="dash", line_color="orange",
                              annotation_text=f"Mejor: iter={best_iter}")
            fig_gb.update_layout(xaxis_title="Número de árboles (iteraciones)",
                                  yaxis_title="Accuracy",
                                  title=f"Staged Score — LR={gb_lr}, depth={gb_depth}",
                                  height=380, margin=dict(t=40))
            st.plotly_chart(fig_gb, use_container_width=True)

        col_gb = st.columns(4)
        col_gb[0].metric("Accuracy (Test)", f"{max(staged_test):.4f}")
        col_gb[1].metric("Mejor iteración", best_iter)
        col_gb[2].metric("Accuracy final", f"{staged_test[-1]:.4f}")
        col_gb[3].metric("Árboles totales", gb_n)

        if max(staged_test) > staged_test[-1] + 0.01:
            st.markdown(f"""
            <div class="warning-box">
            ⚠️ <b>Posible overfitting:</b> El mejor desempeño fue en la iteración {best_iter},
            pero continuó entrenando hasta {gb_n}. Considera usar <code>early stopping</code>
            o reducir el <code>learning_rate</code>.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### ⚡ Comparación: Árbol, Bagging, AdaBoost, Gradient Boosting")
        comp_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris"], key="comp_ds")
        comp_n = st.slider("Número de estimadores", 10, 200, 100, key="comp_n")

        if comp_dataset == "Breast Cancer":
            data = load_breast_cancer()
        else:
            data = load_iris()
        X_comp, y_comp = data.data, data.target
        X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_comp, y_comp, test_size=0.3, random_state=42, stratify=y_comp)

        models_comp = {
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Bagging": BaggingClassifier(n_estimators=comp_n, random_state=42, n_jobs=-1),
            "Random Forest": RandomForestClassifier(n_estimators=comp_n, random_state=42, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=comp_n, random_state=42, algorithm='SAMME'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=comp_n, random_state=42),
        }

        results_comp = []
        for name, m in models_comp.items():
            m.fit(X_tr_c, y_tr_c)
            y_pred_c = m.predict(X_te_c)
            cv_scores = cross_val_score(m, X_comp, y_comp, cv=5, scoring='accuracy', n_jobs=-1)
            results_comp.append({
                'Modelo': name,
                'Accuracy Test': accuracy_score(y_te_c, y_pred_c),
                'F1 (macro)': f1_score(y_te_c, y_pred_c, average='macro'),
                'CV Mean': cv_scores.mean(),
                'CV Std': cv_scores.std(),
            })

        df_comp = pd.DataFrame(results_comp).sort_values('Accuracy Test', ascending=False)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(x=df_comp['Modelo'], y=df_comp['Accuracy Test'],
                                   name='Accuracy Test', marker_color='#1a6b8a',
                                   error_y=dict(type='data', array=df_comp['CV Std'].values)))
        fig_comp.add_trace(go.Bar(x=df_comp['Modelo'], y=df_comp['F1 (macro)'],
                                   name='F1 (macro)', marker_color='#e74c3c'))
        fig_comp.update_layout(barmode='group', xaxis_title="Modelo",
                                yaxis_title="Score", yaxis=dict(range=[0.85, 1.0]),
                                title="Comparación de Modelos", height=380, margin=dict(t=40))
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(df_comp.set_index('Modelo').round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════
#  MODULE 9: MODEL COMPARISON
# ═══════════════════════════════════════════════════════

def show_comparison():
    st.markdown('<div class="section-title">🏆 Comparación Completa de Modelos</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🎯 Benchmark Interactivo", "📊 Radar Chart de Métricas"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            bench_dataset = st.selectbox("Dataset", ["Breast Cancer", "Iris", "Sintético desbalanceado"], key="bench_ds")
            bench_cv = st.slider("K-Fold CV", 3, 10, 5, key="bench_cv")
            bench_trees = st.slider("Árboles en ensambles", 20, 200, 100, key="bench_trees")

        if bench_dataset == "Breast Cancer":
            data = load_breast_cancer(); cn = data.target_names
            X_b, y_b = data.data, data.target
        elif bench_dataset == "Iris":
            data = load_iris(); cn = data.target_names
            X_b, y_b = data.data, data.target
        else:
            X_b, y_b = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)
            cn = ["Clase 0", "Clase 1"]

        X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)

        bench_models = {
            "DT (depth=3)": DecisionTreeClassifier(max_depth=3, random_state=42),
            "DT (depth=10)": DecisionTreeClassifier(max_depth=10, random_state=42),
            "DT (full)": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=bench_trees, random_state=42, n_jobs=-1),
            "AdaBoost": AdaBoostClassifier(n_estimators=bench_trees, random_state=42, algorithm='SAMME'),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=bench_trees, random_state=42),
        }

        bench_results = []
        with st.spinner("Entrenando y evaluando modelos..."):
            for name, m in bench_models.items():
                m.fit(X_tr_b, y_tr_b)
                y_pred_b = m.predict(X_te_b)
                cv_sc = cross_val_score(m, X_b, y_b, cv=bench_cv, scoring='accuracy', n_jobs=-1)
                bench_results.append({
                    'Modelo': name,
                    'Acc. Test': round(accuracy_score(y_te_b, y_pred_b), 4),
                    'F1 (macro)': round(f1_score(y_te_b, y_pred_b, average='macro'), 4),
                    'CV Mean': round(cv_sc.mean(), 4),
                    'CV Std': round(cv_sc.std(), 4),
                    'Overfitting': round(m.score(X_tr_b, y_tr_b) - m.score(X_te_b, y_te_b), 4),
                })

        df_bench = pd.DataFrame(bench_results).sort_values('CV Mean', ascending=False)

        with col2:
            fig_b = px.scatter(df_bench, x='CV Std', y='CV Mean', text='Modelo',
                                size='Acc. Test', color='Overfitting',
                                color_continuous_scale='RdYlGn_r',
                                title='Accuracy CV Mean vs Std (burbuja = Acc. Test)',
                                labels={'CV Std': 'Inestabilidad (Std)', 'CV Mean': 'Accuracy Promedio (CV)'})
            fig_b.update_traces(textposition='top center', textfont_size=10)
            fig_b.update_layout(height=400, margin=dict(t=50, b=40))
            st.plotly_chart(fig_b, use_container_width=True)

        st.dataframe(df_bench.set_index('Modelo'), use_container_width=True)
        best = df_bench.iloc[0]['Modelo']
        st.markdown(f"""
        <div class="success-box">
        🏆 <b>Mejor modelo según CV:</b> {best} con CV Mean={df_bench.iloc[0]['CV Mean']:.4f} ± {df_bench.iloc[0]['CV Std']:.4f}
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📊 Radar Chart de Métricas por Modelo")
        if 'bench_results' not in dir():
            data = load_breast_cancer()
            X_b, y_b = data.data, data.target
            X_tr_b, X_te_b, y_tr_b, y_te_b = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)
            bench_models2 = {
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            }
            radar_results = []
            for name, m in bench_models2.items():
                m.fit(X_tr_b, y_tr_b); y_pred = m.predict(X_te_b)
                proba = m.predict_proba(X_te_b)[:, 1]
                radar_results.append({'Modelo': name,
                    'Accuracy': accuracy_score(y_te_b, y_pred),
                    'Precision': precision_score(y_te_b, y_pred),
                    'Recall': recall_score(y_te_b, y_pred),
                    'F1': f1_score(y_te_b, y_pred),
                    'ROC-AUC': roc_auc_score(y_te_b, proba),
                })
        else:
            radar_results = bench_results

        categories = ['Acc. Test', 'F1 (macro)', 'CV Mean']
        fig_radar = go.Figure()
        colors_radar = ['#1a6b8a', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
        for i, row in enumerate(bench_results if 'bench_results' in dir() else radar_results):
            name = row['Modelo']
            values_r = [row.get('Acc. Test', row.get('Accuracy', 0)),
                         row.get('F1 (macro)', row.get('F1', 0)),
                         row.get('CV Mean', row.get('ROC-AUC', 0))]
            fig_radar.add_trace(go.Scatterpolar(r=values_r + [values_r[0]],
                                                  theta=categories + [categories[0]],
                                                  name=name, fill='toself', opacity=0.4,
                                                  line=dict(color=colors_radar[i % len(colors_radar)])))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.7, 1.0])),
                                 title="Radar de Desempeño", height=450)
        st.plotly_chart(fig_radar, use_container_width=True)


# ═══════════════════════════════════════════════════════
#  MODULE 10: FREE LAB
# ═══════════════════════════════════════════════════════

def show_lab():
    st.markdown('<div class="section-title">🔬 Laboratorio Libre</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="concept-box">
    En este laboratorio puedes <b>experimentar libremente</b> con cualquier combinación de modelos,
    datasets y parámetros. Diseñado para practicar lo aprendido en clase.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        lab_model = st.selectbox("Modelo", [
            "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting", "Bagging"
        ], key="lab_model")
    with col2:
        lab_dataset = st.selectbox("Dataset", [
            "Breast Cancer", "Iris", "Moons", "Circles",
            "Sintético balanceado", "Sintético desbalanceado"
        ], key="lab_ds")
    with col3:
        lab_test_size = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, key="lab_ts")

    # Load dataset
    if lab_dataset == "Breast Cancer":
        data = load_breast_cancer(); X_lab, y_lab = data.data, data.target
    elif lab_dataset == "Iris":
        data = load_iris(); X_lab, y_lab = data.data, data.target
    elif lab_dataset == "Moons":
        X_lab, y_lab = make_moons(n_samples=500, noise=0.25, random_state=42)
    elif lab_dataset == "Circles":
        X_lab, y_lab = make_circles(n_samples=500, noise=0.15, factor=0.5, random_state=42)
    elif lab_dataset == "Sintético balanceado":
        X_lab, y_lab = make_classification(n_samples=1000, n_features=10, random_state=42)
    else:
        X_lab, y_lab = make_classification(n_samples=1000, n_features=10, weights=[0.85, 0.15], random_state=42)

    # Model params
    st.markdown("### ⚙️ Configura los parámetros")
    param_cols = st.columns(4)

    if lab_model == "Decision Tree":
        with param_cols[0]: p_criterion = st.selectbox("criterion", ["gini", "entropy"], key="lab_p1")
        with param_cols[1]: p_max_depth = st.slider("max_depth", 1, 20, 5, key="lab_p2")
        with param_cols[2]: p_mss = st.slider("min_samples_split", 2, 50, 2, key="lab_p3")
        with param_cols[3]: p_msl = st.slider("min_samples_leaf", 1, 30, 1, key="lab_p4")
        model_lab = DecisionTreeClassifier(criterion=p_criterion, max_depth=p_max_depth,
                                            min_samples_split=p_mss, min_samples_leaf=p_msl, random_state=42)
    elif lab_model == "Random Forest":
        with param_cols[0]: p_n = st.slider("n_estimators", 10, 300, 100, key="lab_p1")
        with param_cols[1]: p_md = st.slider("max_depth", 1, 20, 5, key="lab_p2")
        with param_cols[2]: p_mf = st.selectbox("max_features", ["sqrt", "log2"], key="lab_p3")
        with param_cols[3]: p_cw = st.selectbox("class_weight", [None, "balanced"], key="lab_p4")
        model_lab = RandomForestClassifier(n_estimators=p_n, max_depth=p_md, max_features=p_mf,
                                            class_weight=p_cw, random_state=42, n_jobs=-1)
    elif lab_model == "AdaBoost":
        with param_cols[0]: p_n = st.slider("n_estimators", 10, 300, 100, key="lab_p1")
        with param_cols[1]: p_lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.5, 1.0], value=1.0, key="lab_p2")
        model_lab = AdaBoostClassifier(n_estimators=p_n, learning_rate=p_lr, random_state=42, algorithm='SAMME')
    elif lab_model == "Gradient Boosting":
        with param_cols[0]: p_n = st.slider("n_estimators", 10, 300, 100, key="lab_p1")
        with param_cols[1]: p_lr = st.select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.5], value=0.1, key="lab_p2")
        with param_cols[2]: p_md = st.slider("max_depth", 1, 10, 3, key="lab_p3")
        with param_cols[3]: p_sub = st.slider("subsample", 0.3, 1.0, 1.0, 0.1, key="lab_p4")
        model_lab = GradientBoostingClassifier(n_estimators=p_n, learning_rate=p_lr, max_depth=p_md,
                                                subsample=p_sub, random_state=42)
    else:  # Bagging
        with param_cols[0]: p_n = st.slider("n_estimators", 5, 200, 50, key="lab_p1")
        with param_cols[1]: p_ms = st.slider("max_samples", 0.3, 1.0, 1.0, 0.1, key="lab_p2")
        with param_cols[2]: p_mf2 = st.slider("max_features (ratio)", 0.3, 1.0, 1.0, 0.1, key="lab_p3")
        model_lab = BaggingClassifier(n_estimators=p_n, max_samples=p_ms, max_features=p_mf2,
                                       random_state=42, n_jobs=-1)

    if st.button("🚀 Entrenar y Evaluar", type="primary"):
        with st.spinner("Entrenando..."):
            X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(X_lab, y_lab, test_size=lab_test_size, random_state=42, stratify=y_lab)
            model_lab.fit(X_tr_l, y_tr_l)
            y_pred_l = model_lab.predict(X_te_l)
            cv_sc_l = cross_val_score(model_lab, X_lab, y_lab, cv=5, scoring='accuracy', n_jobs=-1)

        col_r = st.columns(5)
        col_r[0].metric("Accuracy (Train)", f"{model_lab.score(X_tr_l, y_tr_l):.4f}")
        col_r[1].metric("Accuracy (Test)", f"{accuracy_score(y_te_l, y_pred_l):.4f}")
        col_r[2].metric("F1 (macro)", f"{f1_score(y_te_l, y_pred_l, average='macro'):.4f}")
        col_r[3].metric("CV 5-Fold Mean", f"{cv_sc_l.mean():.4f}")
        col_r[4].metric("CV 5-Fold Std", f"{cv_sc_l.std():.4f}")

        fig_lab, ax_lab = plt.subplots(figsize=(5, 4))
        cm_l = confusion_matrix(y_te_l, y_pred_l)
        sns.heatmap(cm_l, annot=True, fmt='d', cmap='Blues', ax=ax_lab)
        ax_lab.set_title(f"Matriz de Confusión — {lab_model}", fontsize=12)
        ax_lab.set_xlabel("Predicho"); ax_lab.set_ylabel("Real")
        st.pyplot(fig_lab)
        plt.close()

        report_l = classification_report(y_te_l, y_pred_l)
        st.code(report_l, language="text")

        overfitting_gap = model_lab.score(X_tr_l, y_tr_l) - accuracy_score(y_te_l, y_pred_l)
        if overfitting_gap > 0.1:
            st.markdown(f'<div class="warning-box">⚠️ Gap train-test = {overfitting_gap:.3f} → posible OVERFITTING</div>', unsafe_allow_html=True)
        elif accuracy_score(y_te_l, y_pred_l) < 0.7:
            st.markdown(f'<div class="warning-box">🔵 Accuracy baja → posible UNDERFITTING</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ Buen ajuste (gap={overfitting_gap:.3f})</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MAIN ROUTER
# ═══════════════════════════════════════════════════════

if module == "home":
    show_home()
elif module == "bias_variance":
    show_bias_variance()
elif module == "decision_trees":
    show_decision_trees()
elif module == "entropy":
    show_entropy()
elif module == "metrics":
    show_metrics()
elif module == "validation":
    show_validation()
elif module == "imbalanced":
    show_imbalanced()
elif module == "bagging":
    show_bagging()
elif module == "boosting":
    show_boosting()
elif module == "comparison":
    show_comparison()
elif module == "lab":
    show_lab()
