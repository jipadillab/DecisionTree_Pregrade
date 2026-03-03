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

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               BaggingClassifier, AdaBoostClassifier)
from sklearn.model_selection import (cross_val_score, KFold, learning_curve,
                                      validation_curve, train_test_split)
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                              precision_score, recall_score, f1_score, roc_auc_score,
                              roc_curve, precision_recall_curve)
from sklearn.datasets import (make_classification, make_moons, make_circles,
                               load_iris, load_breast_cancer, load_wine)
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
# THEME-ADAPTIVE CSS
# All coloured boxes use rgba() with low opacity so text
# inherits Streamlit's own colour token (white in dark mode,
# dark in light mode).  Only the header forces white text
# because it sits on a solid dark background.
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a6b8a 0%, #0d4f6b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; font-weight: 700; color: #ffffff !important; }
    .main-header p  { font-size: 0.95rem; margin: 0.3rem 0 0; opacity: 0.9; color: #ffffff !important; }

    .concept-box {
        background: rgba(26, 107, 138, 0.15);
        border-left: 5px solid #1a9fd4;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .formula-box {
        background: rgba(240, 180, 41, 0.15);
        border: 1.5px solid #f0b429;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-family: monospace;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.15);
        border-left: 5px solid #ffc107;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .success-box {
        background: rgba(40, 167, 69, 0.15);
        border-left: 5px solid #28a745;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .metric-card {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .metric-card strong { color: #1a9fd4; }
    .section-title {
        color: #1a9fd4;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid #1a9fd4;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB — transparent backgrounds so plots
# look correct on both dark and light themes.
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "savefig.facecolor": "none",
    "axes.edgecolor":    "#888888",
    "grid.color":        "#888888",
    "grid.alpha":        0.2,
})

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌳 ML Trees & Ensembles")
    st.markdown("**Universidad EAFIT**")
    st.markdown("---")

    MODULES = {
        "🏠 Inicio":                              "home",
        "📊 1. Underfitting & Overfitting":        "bias_variance",
        "🌲 2. Árboles de Decisión":               "decision_trees",
        "📐 3. Entropía & Information Gain":        "entropy",
        "📏 4. Métricas de Desempeño":              "metrics",
        "✅ 5. Validación & Cross-Validation":      "validation",
        "⚖️ 6. Clases Desbalanceadas":             "imbalanced",
        "🌳🌳 7. Bootstrap Aggregating (Bagging)": "bagging",
        "🚀 8. Boosting & Gradient Boosting":      "boosting",
        "🏆 9. Comparación de Modelos":            "comparison",
        "🔬 10. Laboratorio Libre":                "lab",
    }

    selected = st.radio("Módulo:", list(MODULES.keys()), label_visibility="collapsed")
    module   = MODULES[selected]

    st.markdown("---")
    with st.expander("📚 Glosario"):
        st.markdown("""
- **Underfitting** → alto sesgo  
- **Overfitting** → alta varianza  
- **Entropía / Gini** → impureza  
- **Bagging** → paralelo, reduce varianza  
- **Boosting** → secuencial, reduce sesgo  
- **Random Forest** → bagging de árboles  
- **SMOTE** → oversampling sintético  
        """)


# ═══════════════════════════════════════════════════════
#  DATASETS
# ═══════════════════════════════════════════════════════

DATASET_INFO = {
    "🌸 Iris (flores)":              {"desc": "3 especies · 150 muestras · 4 features",        "binary": False},
    "🍷 Wine (vinos)":               {"desc": "3 tipos de vino · 178 muestras · 13 features",  "binary": False},
    "🎗️ Breast Cancer (tumores)":   {"desc": "Maligno vs benigno · 569 muestras · 30 feat.",  "binary": True},
    "🌙 Moons (sintético 2D)":       {"desc": "Dos lunas · 2 features · ideal para fronteras", "binary": True},
    "⭕ Circles (sintético 2D)":     {"desc": "Círculos concéntricos · 2 features",            "binary": True},
    "🎲 Sintético balanceado":        {"desc": "10 features · clases bien separadas",           "binary": True},
    "⚠️ Sintético desbalanceado":    {"desc": "85 % cls 0 · 15 % cls 1 · simula fraude",      "binary": True},
}


def load_dataset(name, n_samples=500, noise=0.2, seed=42):
    """Returns X, y, feature_names, class_names."""
    np.random.seed(seed)
    if name == "🌸 Iris (flores)":
        d = load_iris()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🍷 Wine (vinos)":
        d = load_wine()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🎗️ Breast Cancer (tumores)":
        d = load_breast_cancer()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🌙 Moons (sintético 2D)":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    elif name == "⭕ Circles (sintético 2D)":
        X, y = make_circles(n_samples=n_samples, noise=noise * 0.6, factor=0.5, random_state=seed)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    elif name == "🎲 Sintético balanceado":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, random_state=seed)
        return X, y, [f"Feat_{i}" for i in range(10)], ["Clase 0", "Clase 1"]
    else:
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5,
                                    weights=[0.85, 0.15], random_state=seed)
        return X, y, [f"Feat_{i}" for i in range(10)], ["Cls 0 (may.)", "Cls 1 (min.)"]


# ─────────────────────────────────────────────
# AdaBoost factory — handles API change in sklearn ≥ 1.6
# ─────────────────────────────────────────────
def make_adaboost(n_estimators=100):
    try:
        return AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    except TypeError:
        return AdaBoostClassifier(n_estimators=n_estimators, random_state=42, algorithm='SAMME')


# ─────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────
def plot_decision_boundary(model, X, y, title="", ax=None):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', linewidths=0.5, s=40)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    return fig


def entropy_fn(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def gini_fn(p):
    return 1 - p**2 - (1 - p)**2


PLOTLY_BASE = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')


# ═══════════════════════════════════════════════════════
#  HOME
# ═══════════════════════════════════════════════════════

def show_home():
    st.markdown("""
    <div class="main-header">
        <h1>🌳 Árboles de Decisión & Ensambles</h1>
        <p>Plataforma interactiva · Artificial Intelligence · Universidad EAFIT</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🗺️ Módulos del Curso")
    topics = [
        ("📊","Underfitting & Overfitting","Sesgo, varianza y el trade-off fundamental"),
        ("🌲","Árboles de Decisión","Estructura, splits, profundidad y poda"),
        ("📐","Entropía & Info Gain","Cómo el árbol elige la mejor pregunta"),
        ("📏","Métricas de Desempeño","Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("✅","Validación & CV","Hold-out, K-Fold, Stratified CV"),
        ("⚖️","Clases Desbalanceadas","SMOTE, undersampling, class_weight"),
        ("🌳🌳","Bagging / Random Forest","Ensamble paralelo, feature importance"),
        ("🚀","Boosting","AdaBoost, Gradient Boosting"),
        ("🏆","Comparación de Modelos","Benchmark interactivo con múltiples datasets"),
        ("🔬","Laboratorio Libre","Experimenta con tus propios parámetros"),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(topics):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:1.3rem">{icon}</span>
                <strong> {title}</strong><br>
                <span style="font-size:0.85rem;opacity:0.8">{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Datasets disponibles")
    dc = st.columns(3)
    for i, (name, info) in enumerate(DATASET_INFO.items()):
        tag = "🔵 Binario" if info["binary"] else "🟠 Multi-clase"
        with dc[i % 3]:
            st.markdown(f"""<div class="concept-box">
            <b>{name}</b> <span style="font-size:0.75rem">{tag}</span><br>
            <span style="font-size:0.82rem">{info['desc']}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="concept-box" style="margin-top:1rem">
    💡 <b>Cómo navegar:</b> usa el menú lateral módulo a módulo.
    Cada sección tiene teoría, sliders interactivos, visualizaciones en tiempo real y diagnósticos automáticos.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  1 — BIAS / VARIANCE
# ═══════════════════════════════════════════════════════

def show_bias_variance():
    st.markdown('<div class="section-title">📊 Underfitting, Overfitting & Bias-Variance Tradeoff</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🎮 Árbol & Complejidad", "📉 Curvas de Aprendizaje"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="concept-box">
            <b>🔵 Underfitting</b><br>
            Modelo demasiado simple.<br>• Alto sesgo · baja varianza<br>• Error alto en train Y test
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box">
            <b>🔴 Overfitting</b><br>
            Modelo que memoriza.<br>• Bajo sesgo · alta varianza<br>• Error bajo en train, alto en test
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">
        <b>Error Total = Sesgo² + Varianza + Ruido irreducible</b><br>
        Reducir uno generalmente aumenta el otro → <i>tradeoff</i>
        </div>""", unsafe_allow_html=True)

        complexity = np.linspace(1, 10, 200)
        bias2    = 5 * np.exp(-0.4 * complexity)
        variance = 0.05 * np.exp(0.5 * complexity)
        total    = bias2 + variance + 0.5
        opt_idx  = int(np.argmin(total))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(complexity, bias2,    color='#4da6ff', lw=2.5, label='Sesgo²')
        ax.plot(complexity, variance, color='#ff6b6b', lw=2.5, label='Varianza')
        ax.plot(complexity, total,    color='#aaaaaa', lw=2.5, ls='--', label='Error Total')
        ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Ruido irreducible')
        ax.axvline(complexity[opt_idx], color='#2ecc71', ls='--', label='Óptimo')
        ax.fill_between(complexity[:opt_idx+1], 0, total[:opt_idx+1], alpha=0.08, color='#4da6ff')
        ax.fill_between(complexity[opt_idx:],   0, total[opt_idx:],   alpha=0.08, color='#ff6b6b')
        ax.set_xlabel('Complejidad'); ax.set_ylabel('Error')
        ax.set_title('Tradeoff Sesgo-Varianza', fontweight='bold'); ax.legend(fontsize=9); ax.set_ylim(0, 6)
        st.pyplot(fig); plt.close()

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            ds_bv   = st.selectbox("Dataset", ["🌙 Moons (sintético 2D)","⭕ Circles (sintético 2D)","🌸 Iris (flores)"], key="bv_ds")
            noise_bv= st.slider("Ruido", 0.0, 0.5, 0.2, 0.05, key="bv_n2")
            n_bv    = st.slider("Muestras", 100, 600, 300, 50, key="bv_ns")
            depth_bv= st.slider("max_depth", 1, 15, 3, key="bv_d")
            ts_bv   = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, key="bv_ts")

        X_bv, y_bv, _, _ = load_dataset(ds_bv, n_samples=n_bv, noise=noise_bv)
        X_bv = X_bv[:, :2]
        X_tr, X_te, y_tr, y_te = train_test_split(X_bv, y_bv, test_size=ts_bv, random_state=42, stratify=y_bv)
        m_bv = DecisionTreeClassifier(max_depth=depth_bv, random_state=42).fit(X_tr, y_tr)
        tr_acc = m_bv.score(X_tr, y_tr); te_acc = m_bv.score(X_te, y_te)

        with c2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_decision_boundary(m_bv, X_tr, y_tr, f"Train | {tr_acc:.2%}", axes[0])
            plot_decision_boundary(m_bv, X_te, y_te, f"Test  | {te_acc:.2%}", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

        r1, r2, r3 = st.columns(3)
        gap = tr_acc - te_acc
        if gap > 0.1:    st_lbl, st_col = "🔴 OVERFITTING",   "#dc3545"
        elif te_acc<0.65:st_lbl, st_col = "🔵 UNDERFITTING",  "#007bff"
        else:            st_lbl, st_col = "🟢 Buen ajuste",   "#28a745"
        r1.metric("Train Acc", f"{tr_acc:.2%}")
        r2.metric("Test Acc",  f"{te_acc:.2%}", delta=f"{te_acc-tr_acc:+.2%}")
        r3.markdown(f"<div style='text-align:center;padding:.8rem;border-radius:8px;"
                    f"background:{st_col}22;color:{st_col};font-weight:bold;margin-top:.3rem'>{st_lbl}</div>",
                    unsafe_allow_html=True)

        depths_sw = list(range(1, 16))
        tr_sw = [DecisionTreeClassifier(max_depth=d,random_state=42).fit(X_tr,y_tr).score(X_tr,y_tr) for d in depths_sw]
        te_sw = [DecisionTreeClassifier(max_depth=d,random_state=42).fit(X_tr,y_tr).score(X_te,y_te) for d in depths_sw]
        fig2 = go.Figure([
            go.Scatter(x=depths_sw, y=tr_sw, mode='lines+markers', name='Train', line=dict(color='#e74c3c',width=2)),
            go.Scatter(x=depths_sw, y=te_sw, mode='lines+markers', name='Test',  line=dict(color='#2ecc71',width=2)),
        ])
        fig2.add_vline(x=depth_bv, line_dash="dash", line_color="orange", annotation_text=f"depth={depth_bv}")
        fig2.update_layout(xaxis_title="Profundidad", yaxis_title="Accuracy",
                            height=300, margin=dict(t=20,b=40), **PLOTLY_BASE)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns([1, 2])
        with c1:
            ds_lc   = st.selectbox("Dataset", ["🎗️ Breast Cancer (tumores)","🌸 Iris (flores)","🍷 Wine (vinos)"], key="lc_ds")
            depth_lc= st.slider("max_depth", 1, 15, 5, key="lc_d")
        X_lc, y_lc, _, _ = load_dataset(ds_lc)
        tr_sz, tr_sc, va_sc = learning_curve(
            DecisionTreeClassifier(max_depth=depth_lc,random_state=42),
            X_lc, y_lc, cv=5, train_sizes=np.linspace(0.1,1.0,10), scoring='accuracy', n_jobs=-1)
        with c2:
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(tr_sz, tr_sc.mean(1), 'r-o', lw=2, label='Train')
            ax.fill_between(tr_sz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.12, color='red')
            ax.plot(tr_sz, va_sc.mean(1), 'g-o', lw=2, label='Validación')
            ax.fill_between(tr_sz, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), alpha=0.12, color='green')
            ax.set_xlabel("Tamaño train"); ax.set_ylabel("Accuracy")
            ax.set_title(f"Curvas de Aprendizaje — depth={depth_lc}", fontweight='bold'); ax.legend()
            st.pyplot(fig); plt.close()
        vf = va_sc.mean(1)[-1]; tf = tr_sc.mean(1)[-1]; gap_lc = tf - vf
        if gap_lc > 0.1:
            st.markdown(f'<div class="warning-box">⚠️ <b>OVERFITTING</b> gap={gap_lc:.2%}. Reduce depth.</div>', unsafe_allow_html=True)
        elif vf < 0.75:
            st.markdown(f'<div class="warning-box">🔵 <b>UNDERFITTING</b> val={vf:.2%}. Aumenta complejidad.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ <b>Buen ajuste</b> val={vf:.2%}, gap={gap_lc:.2%}.</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  2 — DECISION TREES
# ═══════════════════════════════════════════════════════

def show_decision_trees():
    st.markdown('<div class="section-title">🌲 Árboles de Decisión</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Conceptos", "🎮 Constructor Interactivo", "🔍 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="concept-box">
            <b>Componentes</b><br>
            • <b>Nodo raíz</b>: primera pregunta<br>
            • <b>Nodo interno</b>: condición sobre una feature<br>
            • <b>Hoja</b>: predicción final<br>
            • <b>Profundidad</b>: niveles raíz → hoja más lejana
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box">
            <b>Hiperparámetros clave</b><br>
            • <code>max_depth</code> → controla overfitting<br>
            • <code>min_samples_split</code> → mínimo para split<br>
            • <code>min_samples_leaf</code> → mínimo en hojas<br>
            • <code>criterion</code> → 'gini' o 'entropy'<br>
            • <code>ccp_alpha</code> → poda
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">
        <b>Algoritmo CART:</b><br>
        1. Para cada feature y umbral → calcular impureza del split<br>
        2. Elegir el split que maximiza el Information Gain<br>
        3. Repetir recursivamente hasta condición de parada
        </div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            dt_ds  = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="dt_ds")
            dt_cr  = st.selectbox("criterion", ["gini","entropy"], key="dt_cr")
            dt_md  = st.slider("max_depth", 1, 12, 3, key="dt_md")
            dt_mss = st.slider("min_samples_split", 2, 50, 2, key="dt_mss")
            dt_msl = st.slider("min_samples_leaf",  1, 30, 1, key="dt_msl")
            dt_ts  = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, key="dt_ts")

        X_dt, y_dt, fn_dt, cn_dt = load_dataset(dt_ds)
        X2 = X_dt[:, :2]
        X_tr, X_te, y_tr, y_te = train_test_split(X2, y_dt, test_size=dt_ts, random_state=42, stratify=y_dt)
        dt = DecisionTreeClassifier(criterion=dt_cr, max_depth=dt_md,
                                     min_samples_split=dt_mss, min_samples_leaf=dt_msl, random_state=42)
        dt.fit(X_tr, y_tr); yp = dt.predict(X_te)

        with c2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_decision_boundary(dt, X_tr, y_tr, "Train", axes[0])
            plot_decision_boundary(dt, X_te, y_te, f"Test Acc={accuracy_score(y_te,yp):.2%}", axes[1])
            plt.tight_layout(); st.pyplot(fig); plt.close()

        mc = st.columns(4)
        mc[0].metric("Accuracy (Test)",  f"{accuracy_score(y_te,yp):.3f}")
        mc[1].metric("Nodos",            dt.tree_.node_count)
        mc[2].metric("Profundidad real", dt.get_depth())
        mc[3].metric("Hojas",            dt.get_n_leaves())

        st.markdown("#### 🌳 Árbol visual")
        fig_t, ax_t = plt.subplots(figsize=(max(10, dt_md*3.5), max(5, dt_md*2)))
        plot_tree(dt, feature_names=fn_dt[:2], class_names=[str(c) for c in cn_dt],
                  filled=True, rounded=True, fontsize=8, ax=ax_t)
        ax_t.set_title(f"criterion={dt_cr}, max_depth={dt_md}", fontweight='bold')
        st.pyplot(fig_t); plt.close()

    with tab3:
        fi_ds = st.selectbox("Dataset", ["🎗️ Breast Cancer (tumores)","🍷 Wine (vinos)","🌸 Iris (flores)"], key="fi_ds")
        fi_d  = st.slider("max_depth", 1, 12, 4, key="fi_d2")
        X_fi, y_fi, fn_fi, _ = load_dataset(fi_ds)
        X_tr_fi, X_te_fi, y_tr_fi, _ = train_test_split(X_fi, y_fi, test_size=0.3, random_state=42, stratify=y_fi)
        dt_fi = DecisionTreeClassifier(max_depth=fi_d, random_state=42).fit(X_tr_fi, y_tr_fi)
        imp   = dt_fi.feature_importances_
        sidx  = np.argsort(imp)[::-1][:15]
        fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
        ax_fi.barh(range(len(sidx)), imp[sidx], color=['#1a9fd4' if i==sidx[0] else '#5ba3c9' for i in sidx])
        ax_fi.set_yticks(range(len(sidx))); ax_fi.set_yticklabels([fn_fi[i] for i in sidx], fontsize=9)
        ax_fi.set_xlabel("Importancia (Gini Reduction)"); ax_fi.set_title("Feature Importance", fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_fi); plt.close()
        rules = export_text(dt_fi, feature_names=list(fn_fi), max_depth=3)
        st.code(rules[:2000]+("\n..." if len(rules)>2000 else ""), language="text")


# ═══════════════════════════════════════════════════════
#  3 — ENTROPY
# ═══════════════════════════════════════════════════════

def show_entropy():
    st.markdown('<div class="section-title">📐 Entropía & Information Gain</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🧮 Calculadora", "🎮 Gini vs Entropía"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="formula-box">
            <b>Entropía de Shannon:</b><br>
            H(π) = −Σ π·log₂(π)<br>
            Binaria: H = −πₚlog₂πₚ − πₙlog₂πₙ<br>
            H=0 → puro · H=1 → máxima mezcla
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="formula-box">
            <b>Gini:</b> 1 − Σπᵢ² = 2p(1−p)<br><br>
            <b>Expected Entropy tras split A:</b><br>
            EH(A) = Σᵢ [(pᵢ+nᵢ)/(p+n)] · H(πᵢ)<br><br>
            <b>Information Gain:</b><br>
            IG(A) = H(padre) − EH(A)  ↑ max
            </div>""", unsafe_allow_html=True)
        p_vals = np.linspace(0.001, 0.999, 200)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(p_vals, [entropy_fn(p) for p in p_vals], color='#4da6ff', lw=2.5, label='Entropía H(p)')
        ax.plot(p_vals, [gini_fn(p)   for p in p_vals], color='#ff6b6b', lw=2.5, label='Gini 2p(1−p)')
        ax.set_xlabel("p (clase positiva)"); ax.set_ylabel("Impureza")
        ax.set_title("Entropía vs Gini", fontweight='bold'); ax.legend()
        ax.axvline(0.5, color='gray', ls='--', alpha=0.5)
        st.pyplot(fig); plt.close()

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Nodo Padre**")
            pp = st.number_input("Positivos", 1, 200, 6, key="ig_pp")
            pn = st.number_input("Negativos", 1, 200, 6, key="ig_pn")
        with c2:
            st.markdown("**Hijo Izquierdo**")
            lp = st.number_input("Positivos", 0, 200, 4, key="ig_lp")
            ln = st.number_input("Negativos", 0, 200, 0, key="ig_ln")
        with c3:
            st.markdown("**Hijo Derecho**")
            rp = st.number_input("Positivos", 0, 200, 2, key="ig_rp")
            rn = st.number_input("Negativos", 0, 200, 6, key="ig_rn")

        tot = pp+pn
        h_p = entropy_fn(pp/tot) if tot>0 else 0
        tl, tr2 = lp+ln, rp+rn
        hl = entropy_fn(lp/tl) if tl>0 else 0
        hr = entropy_fn(rp/tr2) if tr2>0 else 0
        eh = (tl/tot)*hl + (tr2/tot)*hr if tot>0 else 0
        ig = h_p - eh

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("H(Padre)", f"{h_p:.4f}"); r2.metric("H(Izq)", f"{hl:.4f}")
        r3.metric("EH(Split)",f"{eh:.4f}");  r4.metric("IG",     f"{ig:.4f}", delta="↑ max")

        fig_b, ax_b = plt.subplots(figsize=(7,3))
        ax_b.bar(['H(Padre)','H(Izq)','H(Der)','EH(Split)','IG'],
                  [h_p, hl, hr, eh, ig],
                  color=['#1a9fd4','#5bc0de','#5bc0de','#e74c3c','#2ecc71'])
        ax_b.set_title(f"Information Gain = {ig:.4f}", fontweight='bold')
        st.pyplot(fig_b); plt.close()

    with tab3:
        gc_ds = st.selectbox("Dataset", ["🌸 Iris (flores)","🎗️ Breast Cancer (tumores)","🍷 Wine (vinos)"], key="gc_ds")
        gc_d  = st.slider("max_depth", 1, 8, 3, key="gc_d")
        X_gc, y_gc, fn_gc, cn_gc = load_dataset(gc_ds)
        X2 = X_gc[:, :2]
        X_tr, X_te, y_tr, y_te = train_test_split(X2, y_gc, test_size=0.3, random_state=42, stratify=y_gc)
        dtg = DecisionTreeClassifier(criterion='gini',    max_depth=gc_d, random_state=42).fit(X_tr, y_tr)
        dte = DecisionTreeClassifier(criterion='entropy', max_depth=gc_d, random_state=42).fit(X_tr, y_tr)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_decision_boundary(dtg, X_te, y_te, f"Gini — {dtg.score(X_te,y_te):.2%}", axes[0])
        plot_decision_boundary(dte, X_te, y_te, f"Entropy — {dte.score(X_te,y_te):.2%}", axes[1])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown("""<div class="concept-box">
        💡 En la mayoría de casos Gini y Entropía dan resultados similares.
        Gini es más rápido (sin log). Entropía puede producir árboles más balanceados.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  4 — METRICS
# ═══════════════════════════════════════════════════════

def show_metrics():
    st.markdown('<div class="section-title">📏 Métricas de Desempeño</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Matriz de Confusión", "📈 ROC & PR Curves", "🧮 Calculadora"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="formula-box">
            <b>Métricas binarias:</b><br>
            • Accuracy  = (TP+TN)/Total<br>
            • Precision = TP/(TP+FP)<br>
            • Recall    = TP/(TP+FN)<br>
            • F1        = 2·P·R/(P+R)<br>
            • ROC-AUC   = área bajo curva ROC
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="warning-box">
            ⚠️ <b>¿Cuándo NO usar solo Accuracy?</b><br>
            Con clases desbalanceadas, un modelo que siempre predice la clase mayoritaria
            puede tener Accuracy=95 % pero Recall=0 % para la minoritaria.
            </div>""", unsafe_allow_html=True)

        cm_ds = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="cm_ds")
        cm_d  = st.slider("max_depth", 1, 15, 4, key="cm_d")
        X_cm, y_cm, _, cn_cm = load_dataset(cm_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_cm, y_cm, test_size=0.3, random_state=42, stratify=y_cm)
        m_cm = DecisionTreeClassifier(max_depth=cm_d, random_state=42).fit(X_tr, y_tr)
        yp_cm = m_cm.predict(X_te)
        cm_arr = confusion_matrix(y_te, yp_cm)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=cn_cm, yticklabels=cn_cm, linewidths=0.5)
        ax_cm.set_xlabel("Predicho"); ax_cm.set_ylabel("Real")
        ax_cm.set_title("Matriz de Confusión", fontweight='bold')
        st.pyplot(fig_cm); plt.close()

        n_cls = len(np.unique(y_cm))
        if n_cls == 2:
            mc = st.columns(5)
            mc[0].metric("Accuracy",  f"{accuracy_score(y_te,yp_cm):.3f}")
            mc[1].metric("Precision", f"{precision_score(y_te,yp_cm):.3f}")
            mc[2].metric("Recall",    f"{recall_score(y_te,yp_cm):.3f}")
            mc[3].metric("F1",        f"{f1_score(y_te,yp_cm):.3f}")
            mc[4].metric("ROC-AUC",   f"{roc_auc_score(y_te, m_cm.predict_proba(X_te)[:,1]):.3f}")
        st.code(classification_report(y_te, yp_cm, target_names=[str(c) for c in cn_cm]), language="text")

    with tab2:
        roc_ds = st.selectbox("Dataset (binario)", ["🎗️ Breast Cancer (tumores)","🎲 Sintético balanceado","⚠️ Sintético desbalanceado"], key="roc_ds")
        X_roc, y_roc, _, _ = load_dataset(roc_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_roc, y_roc, test_size=0.3, random_state=42, stratify=y_roc)

        fig_roc = make_subplots(rows=1, cols=2, subplot_titles=["ROC Curve","Precision-Recall"])
        for d, lbl, col in zip([2,4,6,None],['d=2','d=4','d=6','full'],
                                 ['#e74c3c','#f39c12','#2ecc71','#3498db']):
            m_ = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
            proba = m_.predict_proba(X_te)[:,1]
            fpr, tpr, _ = roc_curve(y_te, proba)
            auc = roc_auc_score(y_te, proba)
            prec, rec, _ = precision_recall_curve(y_te, proba)
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr, name=f"{lbl}(AUC={auc:.2f})", line=dict(color=col,width=2)), row=1,col=1)
            fig_roc.add_trace(go.Scatter(x=rec,y=prec, name=lbl, showlegend=False, line=dict(color=col,width=2,dash='dot')), row=1,col=2)
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='gray')),row=1,col=1)
        fig_roc.update_layout(height=420, **PLOTLY_BASE)
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            tp = st.number_input("TP", 0, 1000, 85); fp = st.number_input("FP", 0, 1000, 10)
        with c2:
            fn = st.number_input("FN", 0, 1000, 15); tn = st.number_input("TN", 0, 1000, 90)
        tot_c = tp+fp+fn+tn
        if tot_c > 0:
            acc_c  = (tp+tn)/tot_c
            prec_c = tp/(tp+fp) if tp+fp>0 else 0
            rec_c  = tp/(tp+fn) if tp+fn>0 else 0
            spec_c = tn/(tn+fp) if tn+fp>0 else 0
            f1_c   = 2*prec_c*rec_c/(prec_c+rec_c) if prec_c+rec_c>0 else 0
            rc = st.columns(3)
            rc[0].metric("Accuracy",    f"{acc_c:.3f}")
            rc[1].metric("Precision",   f"{prec_c:.3f}")
            rc[2].metric("Recall",      f"{rec_c:.3f}")
            rc2 = st.columns(3)
            rc2[0].metric("Specificity",f"{spec_c:.3f}")
            rc2[1].metric("F1-Score",   f"{f1_c:.3f}")
            rc2[2].metric("FPR",        f"{1-spec_c:.3f}")
            fig_c, ax_c = plt.subplots(figsize=(4, 3))
            sns.heatmap([[tn,fp],[fn,tp]], annot=True, fmt='d', cmap='Blues', ax=ax_c,
                        xticklabels=['Pred−','Pred+'], yticklabels=['Real−','Real+'])
            st.pyplot(fig_c); plt.close()


# ═══════════════════════════════════════════════════════
#  5 — VALIDATION
# ═══════════════════════════════════════════════════════

def show_validation():
    st.markdown('<div class="section-title">✅ Validación & Cross-Validation</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Estrategias", "🎮 K-Fold Interactivo", "🔍 Validation Curve"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="concept-box">
            <b>1. Hold-Out</b> — Divide en train/test (ej. 70/30).<br>
            ✅ Rápido · ❌ Alta varianza en la estimación
            </div>
            <div class="concept-box">
            <b>2. K-Fold Cross-Validation</b> — K particiones, K evaluaciones.<br>
            ✅ Estimación robusta · ❌ K× más costoso
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box">
            <b>3. Stratified K-Fold</b> — mantiene proporción de clases.<br>
            ✅ Esencial para datos desbalanceados
            </div>
            <div class="concept-box">
            <b>4. Leave-One-Out</b> — K = N.<br>
            ✅ Máximo uso de datos · ❌ Muy costoso
            </div>""", unsafe_allow_html=True)

        K = 5
        fig_kf, ax_kf = plt.subplots(figsize=(10, 3))
        for fold in range(K):
            for part in range(K):
                col_kf = '#e74c3c' if part==fold else '#2ecc71'
                lbl    = 'Val' if part==fold else 'Tr'
                rect   = mpatches.Rectangle([part/K,(K-fold-1)/K+0.05], 1/K-0.01, 0.8/K, color=col_kf, alpha=0.75)
                ax_kf.add_patch(rect)
                ax_kf.text(part/K+0.5/K, (K-fold-1)/K+0.05+0.4/K, lbl, ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            ax_kf.text(-0.02, (K-fold-1)/K+0.05+0.4/K, f'Fold {fold+1}', ha='right', va='center', fontsize=9)
        ax_kf.set_xlim(-0.15,1.05); ax_kf.set_ylim(0,1.1); ax_kf.axis('off')
        ax_kf.set_title("K-Fold Cross-Validation (K=5)", fontweight='bold')
        ax_kf.legend(handles=[mpatches.Patch(color='#2ecc71',alpha=0.75,label='Train'),
                               mpatches.Patch(color='#e74c3c',alpha=0.75,label='Validación')], loc='lower right')
        st.pyplot(fig_kf); plt.close()

    with tab2:
        c1, c2 = st.columns([1,2])
        with c1:
            cv_ds  = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="cv_ds")
            cv_d   = st.slider("max_depth", 1, 15, 4, key="cv_d")
            k_f    = st.slider("K folds", 3, 20, 5, key="cv_k")
        X_cv, y_cv, _, _ = load_dataset(cv_ds)
        sc_cv = cross_val_score(DecisionTreeClassifier(max_depth=cv_d,random_state=42),
                                 X_cv, y_cv, cv=KFold(k_f, shuffle=True, random_state=42), scoring='accuracy')
        with c2:
            fig_cv, ax_cv = plt.subplots(figsize=(9, 4))
            folds_cv = list(range(1, k_f+1))
            cols_cv  = ['#e74c3c' if s<sc_cv.mean()-sc_cv.std() else
                         '#2ecc71' if s>sc_cv.mean()+sc_cv.std() else '#3498db' for s in sc_cv]
            bars_cv = ax_cv.bar(folds_cv, sc_cv, color=cols_cv)
            ax_cv.axhline(sc_cv.mean(), ls='--', lw=2, label=f'Media={sc_cv.mean():.3f}')
            ax_cv.axhspan(sc_cv.mean()-sc_cv.std(), sc_cv.mean()+sc_cv.std(), alpha=0.1, color='gray',
                           label=f'±1σ={sc_cv.std():.3f}')
            ax_cv.set_xlabel("Fold"); ax_cv.set_ylabel("Accuracy")
            ax_cv.set_title(f"{k_f}-Fold CV — {cv_ds}", fontweight='bold')
            ax_cv.set_xticks(folds_cv); ax_cv.legend(fontsize=9)
            ax_cv.set_ylim(max(0, sc_cv.min()-0.1), min(1.05, sc_cv.max()+0.05))
            for bar, sc in zip(bars_cv, sc_cv):
                ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{sc:.3f}',
                            ha='center', va='bottom', fontsize=8)
            st.pyplot(fig_cv); plt.close()
        cc = st.columns(4)
        cc[0].metric("Media",  f"{sc_cv.mean():.4f}")
        cc[1].metric("Std",    f"{sc_cv.std():.4f}")
        cc[2].metric("Mínimo", f"{sc_cv.min():.4f}")
        cc[3].metric("Máximo", f"{sc_cv.max():.4f}")

    with tab3:
        vc_ds  = st.selectbox("Dataset", ["🎗️ Breast Cancer (tumores)","🌸 Iris (flores)","🍷 Wine (vinos)"], key="vc_ds")
        vc_par = st.selectbox("Hiperparámetro", ["max_depth","min_samples_split","min_samples_leaf"], key="vc_p")
        X_vc, y_vc, _, _ = load_dataset(vc_ds)
        pr = np.arange(1,16) if vc_par=='max_depth' else (np.arange(2,50,3) if vc_par=='min_samples_split' else np.arange(1,30,2))
        tr_sc_vc, va_sc_vc = validation_curve(DecisionTreeClassifier(random_state=42), X_vc, y_vc,
                                               param_name=vc_par, param_range=pr, cv=5, scoring='accuracy', n_jobs=-1)
        best_i = int(np.argmax(va_sc_vc.mean(1)))
        fig_vc, ax_vc = plt.subplots(figsize=(9, 4))
        ax_vc.plot(pr, tr_sc_vc.mean(1), 'r-o', lw=2, label='Train')
        ax_vc.fill_between(pr, tr_sc_vc.mean(1)-tr_sc_vc.std(1), tr_sc_vc.mean(1)+tr_sc_vc.std(1), alpha=0.12, color='red')
        ax_vc.plot(pr, va_sc_vc.mean(1), 'g-o', lw=2, label='Validación')
        ax_vc.fill_between(pr, va_sc_vc.mean(1)-va_sc_vc.std(1), va_sc_vc.mean(1)+va_sc_vc.std(1), alpha=0.12, color='green')
        ax_vc.axvline(pr[best_i], color='orange', ls='--', label=f'Mejor: {vc_par}={pr[best_i]}')
        ax_vc.set_xlabel(vc_par); ax_vc.set_ylabel("Accuracy")
        ax_vc.set_title(f"Validation Curve — {vc_par}", fontweight='bold'); ax_vc.legend()
        st.pyplot(fig_vc); plt.close()
        st.markdown(f'<div class="success-box">✅ Mejor: <code>{vc_par}={pr[best_i]}</code> · CV={va_sc_vc.mean(1)[best_i]:.4f}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  6 — IMBALANCED
# ═══════════════════════════════════════════════════════

def show_imbalanced():
    st.markdown('<div class="section-title">⚖️ Clases Desbalanceadas</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📖 Técnicas", "🎮 Comparación"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="concept-box"><b>1. Oversampling</b><br>
            Aumenta clase minoritaria.<br>
            • <b>SMOTE</b>: interpola vecinos cercanos<br>
            • <b>ADASYN</b>: SMOTE adaptativo</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box"><b>2. Undersampling</b><br>
            Reduce clase mayoritaria.<br>
            • <b>Random</b>: elimina aleatoriamente<br>
            • <b>Tomek Links</b>: pares cercanos</div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="concept-box"><b>3. class_weight</b><br>
            Penaliza errores en cls minoritaria.<br>
            <code>class_weight='balanced'</code><br>
            ✅ No modifica el dataset</div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            imb_r  = st.slider("% clase minoritaria", 2, 50, 10)
            n_imb  = st.slider("Muestras totales", 300, 2000, 600, 100)
            tech   = st.selectbox("Técnica", ["Sin balanceo","SMOTE","Undersampling","class_weight='balanced'"])
            imb_d  = st.slider("max_depth", 1, 15, 5, key="imb_d")

        n0 = int(n_imb*(1-imb_r/100)); n1 = int(n_imb*imb_r/100)
        X_i = np.vstack([np.random.randn(n0,2)+[2,2], np.random.randn(n1,2)])
        y_i = np.hstack([np.zeros(n0), np.ones(n1)])
        idx = np.random.permutation(len(y_i)); X_i, y_i = X_i[idx], y_i[idx]
        X_tr, X_te, y_tr, y_te = train_test_split(X_i, y_i, test_size=0.3, random_state=42, stratify=y_i)

        cw = None
        if tech == "SMOTE":
            try: X_tr_b, y_tr_b = SMOTE(random_state=42).fit_resample(X_tr, y_tr)
            except Exception: X_tr_b, y_tr_b = X_tr, y_tr
        elif tech == "Undersampling":
            X_tr_b, y_tr_b = RandomUnderSampler(random_state=42).fit_resample(X_tr, y_tr)
        elif tech == "class_weight='balanced'":
            X_tr_b, y_tr_b = X_tr, y_tr; cw = 'balanced'
        else:
            X_tr_b, y_tr_b = X_tr, y_tr

        m_i = DecisionTreeClassifier(max_depth=imb_d, random_state=42, class_weight=cw).fit(X_tr_b, y_tr_b)
        yp_i = m_i.predict(X_te)

        with c2:
            fig_i, ax_i = plt.subplots(1, 2, figsize=(12, 5))
            vc = pd.Series(y_tr_b).value_counts().sort_index()
            ax_i[0].bar(['Clase 0','Clase 1'], vc.values, color=['#3498db','#e74c3c'])
            ax_i[0].set_title(f"Distribución train\n({tech})", fontweight='bold')
            for j, v in enumerate(vc.values):
                ax_i[0].text(j, v+5, str(v), ha='center', fontweight='bold')
            cm_i = confusion_matrix(y_te, yp_i)
            sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues', ax=ax_i[1],
                        xticklabels=['Pred 0','Pred 1'], yticklabels=['Real 0','Real 1'])
            ax_i[1].set_title("Confusión (Test)", fontweight='bold')
            plt.tight_layout(); st.pyplot(fig_i); plt.close()

        ci = st.columns(4)
        ci[0].metric("Accuracy",         f"{accuracy_score(y_te,yp_i):.3f}")
        ci[1].metric("Precision (cls 1)",f"{precision_score(y_te,yp_i,zero_division=0):.3f}")
        ci[2].metric("Recall (cls 1)",   f"{recall_score(y_te,yp_i,zero_division=0):.3f}", delta="↑ crítico")
        ci[3].metric("F1 (cls 1)",       f"{f1_score(y_te,yp_i,zero_division=0):.3f}")


# ═══════════════════════════════════════════════════════
#  7 — BAGGING
# ═══════════════════════════════════════════════════════

def show_bagging():
    st.markdown('<div class="section-title">🌳🌳 Bootstrap Aggregating & Random Forest</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🌲 Random Forest Interactivo", "📊 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="concept-box">
            <b>Bagging</b><br>
            1. B subconjuntos vía bootstrapping<br>
            2. Modelo en cada subconjunto <b>(paralelo)</b><br>
            3. Promedio (regresión) o votación (clasificación)<br>
            <b>Objetivo:</b> reducir varianza
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box">
            <b>Random Forest = Bagging + Feature Subsampling</b><br>
            En cada split usa solo √n_features aleatorias →
            descorelaciona árboles → menor varianza.<br>
            <b>OOB Score</b>: validación gratuita con el ~37% de muestras no usadas.
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">
        Var(ensemble) = ρσ² + (1−ρ)σ²/B<br>
        RF reduce ρ (correlación entre árboles) al aleatorizar features.
        </div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1,2])
        with c1:
            rf_ds = st.selectbox("Dataset", list(DATASET_INFO.keys()), key="rf_ds")
            rf_n  = st.slider("n_estimators", 1, 200, 50, key="rf_n")
            rf_md = st.slider("max_depth", 1, 20, 5, key="rf_md")
            rf_mf = st.selectbox("max_features", ["sqrt","log2","None (todas)"], key="rf_mf")
            rf_b  = st.checkbox("Bootstrap", value=True, key="rf_b")

        mf_rf = None if rf_mf=="None (todas)" else rf_mf
        X_rf, y_rf, _, _ = load_dataset(rf_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_rf, y_rf, test_size=0.3, random_state=42, stratify=y_rf)
        rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_md, max_features=mf_rf,
                                     bootstrap=rf_b, oob_score=rf_b, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr); yp_rf = rf.predict(X_te)

        with c2:
            n_rng = [n for n in [1,5,10,20,30,50,75,100,150,200] if n<=max(rf_n,20)]
            accs_rf = [RandomForestClassifier(n_estimators=n, max_depth=rf_md, max_features=mf_rf,
                                               bootstrap=rf_b, random_state=42, n_jobs=-1
                                              ).fit(X_tr,y_tr).score(X_te,y_te) for n in n_rng]
            fig_rf = go.Figure([
                go.Scatter(x=n_rng, y=accs_rf, mode='lines+markers', name='Accuracy',
                           line=dict(color='#2ecc71',width=2))
            ])
            fig_rf.add_vline(x=rf_n, line_dash="dash", line_color="orange", annotation_text=f"n={rf_n}")
            fig_rf.update_layout(xaxis_title="Árboles", yaxis_title="Accuracy",
                                  title="Accuracy vs Número de Árboles", height=350, **PLOTLY_BASE)
            st.plotly_chart(fig_rf, use_container_width=True)

        cr = st.columns(4 if rf_b else 3)
        cr[0].metric("Accuracy (Test)", f"{accuracy_score(y_te,yp_rf):.4f}")
        cr[1].metric("F1 (macro)",      f"{f1_score(y_te,yp_rf,average='macro'):.4f}")
        cr[2].metric("Árboles",          rf_n)
        if rf_b: cr[3].metric("OOB Score", f"{rf.oob_score_:.4f}")

        single = DecisionTreeClassifier(max_depth=rf_md, random_state=42).fit(X_tr,y_tr)
        imp_val = accuracy_score(y_te,yp_rf) - single.score(X_te,y_te)
        cls = "success-box" if imp_val>=0 else "warning-box"
        st.markdown(f'<div class="{cls}">🌲 Árbol: <b>{single.score(X_te,y_te):.4f}</b> · '
                    f'🌳🌳 RF: <b>{accuracy_score(y_te,yp_rf):.4f}</b> · Mejora: <b>{imp_val:+.4f}</b></div>',
                    unsafe_allow_html=True)

    with tab3:
        fi_ds_rf = st.selectbox("Dataset", ["🎗️ Breast Cancer (tumores)","🍷 Wine (vinos)","🌸 Iris (flores)"], key="fi_rf_ds")
        fi_n_rf  = st.slider("n_estimators", 10, 300, 100, key="fi_rf_n")
        X_fi, y_fi, fn_fi, _ = load_dataset(fi_ds_rf)
        X_tr_fi, _, y_tr_fi, _ = train_test_split(X_fi, y_fi, test_size=0.3, random_state=42, stratify=y_fi)
        rf_fi = RandomForestClassifier(n_estimators=fi_n_rf, random_state=42, n_jobs=-1).fit(X_tr_fi, y_tr_fi)
        imp_fi  = rf_fi.feature_importances_
        stds_fi = np.std([t.feature_importances_ for t in rf_fi.estimators_], axis=0)
        sidx_fi = np.argsort(imp_fi)[::-1][:15]
        fig_fi_r, ax_fi_r = plt.subplots(figsize=(9,6))
        ax_fi_r.barh(range(len(sidx_fi)), imp_fi[sidx_fi], xerr=stds_fi[sidx_fi],
                      color='#1a9fd4', alpha=0.8, error_kw=dict(ecolor='gray',capsize=3))
        ax_fi_r.set_yticks(range(len(sidx_fi)))
        ax_fi_r.set_yticklabels([fn_fi[i] for i in sidx_fi], fontsize=9)
        ax_fi_r.set_xlabel("Importancia media ± σ")
        ax_fi_r.set_title(f"Feature Importance — RF ({fi_n_rf} árboles)", fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_fi_r); plt.close()


# ═══════════════════════════════════════════════════════
#  8 — BOOSTING
# ═══════════════════════════════════════════════════════

def show_boosting():
    st.markdown('<div class="section-title">🚀 Boosting & Gradient Boosting</div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🎮 Gradient Boosting", "⚡ Comparación Algoritmos"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="concept-box">
            <b>Boosting — Secuencial</b><br>
            1. Entrena modelo₁ con datos originales<br>
            2. Identifica errores<br>
            3. Entrena modelo₂ ponderando más los errores<br>
            4. Repite B veces<br>
            5. Combinación ponderada<br>
            <b>Objetivo:</b> reducir sesgo
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="concept-box">
            <b>Gradient Boosting</b><br>
            Cada árbol predice el <b>residual</b> del ensamble anterior:<br>
            r₁ = y − ŷ₁ → modelo₂ predice r₁<br>
            F(x) = η·Σfₜ(x)  (η = learning rate)<br><br>
            Variantes: AdaBoost, GBM, XGBoost, LightGBM
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="formula-box">
        Bagging → paralelo · reduce varianza<br>
        Boosting → secuencial · reduce sesgo
        </div>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1,2])
        with c1:
            gb_ds  = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="gb_ds")
            gb_n   = st.slider("n_estimators", 10, 300, 100, key="gb_n")
            gb_lr  = st.select_slider("learning_rate", [0.001,0.01,0.05,0.1,0.2,0.5,1.0], value=0.1, key="gb_lr")
            gb_d   = st.slider("max_depth", 1, 10, 3, key="gb_d")
            gb_sub = st.slider("subsample", 0.3, 1.0, 1.0, 0.1, key="gb_sub")

        X_gb, y_gb, _, _ = load_dataset(gb_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_gb, y_gb, test_size=0.3, random_state=42, stratify=y_gb)
        gb_m = GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr,
                                           max_depth=gb_d, subsample=gb_sub, random_state=42).fit(X_tr, y_tr)
        # staged_score was removed in sklearn ≥ 1.4 — compute manually via staged_predict
        staged_tr = [accuracy_score(y_tr, yp) for yp in gb_m.staged_predict(X_tr)]
        staged_te = [accuracy_score(y_te, yp) for yp in gb_m.staged_predict(X_te)]
        best_it   = int(np.argmax(staged_te))

        with c2:
            fig_gb = go.Figure([
                go.Scatter(y=staged_tr, mode='lines', name='Train', line=dict(color='#e74c3c',width=2)),
                go.Scatter(y=staged_te, mode='lines', name='Test',  line=dict(color='#2ecc71',width=2)),
            ])
            fig_gb.add_vline(x=best_it, line_dash="dash", line_color="orange",
                              annotation_text=f"Mejor iter={best_it}")
            fig_gb.update_layout(xaxis_title="Iteraciones", yaxis_title="Accuracy",
                                  title=f"Staged Score — LR={gb_lr}, depth={gb_d}",
                                  height=380, **PLOTLY_BASE)
            st.plotly_chart(fig_gb, use_container_width=True)

        cg = st.columns(4)
        cg[0].metric("Mejor Acc (Test)", f"{max(staged_te):.4f}")
        cg[1].metric("Mejor iter",       best_it)
        cg[2].metric("Acc final",        f"{staged_te[-1]:.4f}")
        cg[3].metric("Árboles",          gb_n)
        if max(staged_te) > staged_te[-1]+0.01:
            st.markdown(f'<div class="warning-box">⚠️ Overfitting en iter={best_it}. Considera early stopping.</div>', unsafe_allow_html=True)

    with tab3:
        cmp_ds = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="cmp_ds")
        cmp_n  = st.slider("n_estimators", 10, 200, 100, key="cmp_n")
        X_cmp, y_cmp, _, _ = load_dataset(cmp_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_cmp, y_cmp, test_size=0.3, random_state=42, stratify=y_cmp)

        models_c = {
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Bagging":        BaggingClassifier(n_estimators=cmp_n, random_state=42, n_jobs=-1),
            "Random Forest":  RandomForestClassifier(n_estimators=cmp_n, random_state=42, n_jobs=-1),
            "AdaBoost":       make_adaboost(cmp_n),
            "Grad. Boosting": GradientBoostingClassifier(n_estimators=cmp_n, random_state=42),
        }
        rows_c = []
        for name, m in models_c.items():
            m.fit(X_tr, y_tr); yp = m.predict(X_te)
            cv = cross_val_score(m, X_cmp, y_cmp, cv=5, scoring='accuracy', n_jobs=-1)
            rows_c.append({'Modelo':name,'Acc. Test':accuracy_score(y_te,yp),
                            'F1 (macro)':f1_score(y_te,yp,average='macro'),
                            'CV Mean':cv.mean(),'CV Std':cv.std()})
        df_c = pd.DataFrame(rows_c).sort_values('Acc. Test', ascending=False)
        fig_c = go.Figure([
            go.Bar(x=df_c['Modelo'], y=df_c['Acc. Test'],  name='Accuracy Test', marker_color='#1a9fd4'),
            go.Bar(x=df_c['Modelo'], y=df_c['F1 (macro)'], name='F1 (macro)',    marker_color='#e74c3c'),
        ])
        lo_y = max(0, df_c['F1 (macro)'].min()-0.05)
        fig_c.update_layout(barmode='group', yaxis=dict(range=[lo_y,1.0]),
                             height=380, **PLOTLY_BASE)
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(df_c.set_index('Modelo').round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════
#  9 — COMPARISON
# ═══════════════════════════════════════════════════════

def show_comparison():
    st.markdown('<div class="section-title">🏆 Comparación Completa de Modelos</div>', unsafe_allow_html=True)
    st.markdown("""<div class="concept-box">
    Benchmark completo con validación cruzada 5-fold.
    Selecciona el dataset y ajusta el número de estimadores.
    </div>""", unsafe_allow_html=True)

    with st.expander("📦 Descripción de los datasets"):
        for name, info in DATASET_INFO.items():
            tag = "🔵 Binario" if info["binary"] else "🟠 Multi-clase"
            st.markdown(f"**{name}** {tag} — {info['desc']}")

    c1, c2, c3 = st.columns(3)
    with c1: bench_ds = st.selectbox("Dataset", list(DATASET_INFO.keys()), key="bench_ds")
    with c2: bench_cv = st.slider("K-Fold CV", 3, 10, 5, key="bench_cv")
    with c3: bench_n  = st.slider("Estimadores en ensambles", 20, 200, 100, key="bench_n")

    X_b, y_b, _, _ = load_dataset(bench_ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)

    bench_models = {
        "DT depth=2":        DecisionTreeClassifier(max_depth=2,    random_state=42),
        "DT depth=5":        DecisionTreeClassifier(max_depth=5,    random_state=42),
        "DT full":           DecisionTreeClassifier(max_depth=None, random_state=42),
        "Bagging":           BaggingClassifier(n_estimators=bench_n, random_state=42, n_jobs=-1),
        "Random Forest":     RandomForestClassifier(n_estimators=bench_n, random_state=42, n_jobs=-1),
        "AdaBoost":          make_adaboost(bench_n),
        "Grad. Boosting":    GradientBoostingClassifier(n_estimators=bench_n, random_state=42),
    }

    bench_rows = []
    with st.spinner("Entrenando y evaluando…"):
        for name, m in bench_models.items():
            m.fit(X_tr, y_tr); yp = m.predict(X_te)
            tr_a = m.score(X_tr, y_tr); te_a = accuracy_score(y_te, yp)
            cv   = cross_val_score(m, X_b, y_b, cv=bench_cv, scoring='accuracy', n_jobs=-1)
            bench_rows.append({
                'Modelo':   name,
                'Train Acc':round(tr_a,4), 'Test Acc': round(te_a,4),
                'F1 (macro)':round(f1_score(y_te,yp,average='macro'),4),
                'CV Mean':  round(cv.mean(),4), 'CV Std': round(cv.std(),4),
                'Gap (overfit)': round(tr_a-te_a,4),
            })

    df_b = pd.DataFrame(bench_rows).sort_values('CV Mean', ascending=False)

    tab_a, tab_b, tab_c = st.tabs(["📊 Tabla", "🫧 Scatter", "🕸️ Radar"])

    with tab_a:
        st.dataframe(
            df_b.set_index('Modelo').style
                .background_gradient(subset=['CV Mean','Test Acc'], cmap='Blues')
                .background_gradient(subset=['Gap (overfit)'], cmap='Reds_r'),
            use_container_width=True)
        best = df_b.iloc[0]
        st.markdown(f'<div class="success-box">🏆 <b>Mejor (CV):</b> {best["Modelo"]} — '
                    f'{best["CV Mean"]:.4f} ± {best["CV Std"]:.4f}</div>', unsafe_allow_html=True)

    with tab_b:
        fig_sc = px.scatter(df_b, x='CV Std', y='CV Mean', text='Modelo', size='Test Acc',
                             color='Gap (overfit)', color_continuous_scale='RdYlGn_r',
                             title='Accuracy CV Mean vs Inestabilidad')
        fig_sc.update_traces(textposition='top center', textfont_size=10)
        fig_sc.update_layout(height=420, **PLOTLY_BASE)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab_c:
        cats = ['Test Acc','F1 (macro)','CV Mean']
        fig_r = go.Figure()
        pal = ['#1a9fd4','#e74c3c','#2ecc71','#f39c12','#9b59b6','#1abc9c','#e67e22']
        for i, (_, row) in enumerate(df_b.iterrows()):
            vals = [row['Test Acc'], row['F1 (macro)'], row['CV Mean']]
            fig_r.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                                             name=row['Modelo'], fill='toself', opacity=0.45,
                                             line=dict(color=pal[i%len(pal)])))
        lo_r = max(0.5, df_b[cats].values.min()-0.05)
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[lo_r,1.0])),
                             title="Radar de Desempeño", height=460,
                             paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Multi-dataset benchmark ──────────────────────────
    st.markdown("---")
    st.markdown("### 🗃️ Benchmark en múltiples datasets")
    multi_ds = st.multiselect(
        "Selecciona datasets:",
        list(DATASET_INFO.keys()),
        default=["🌸 Iris (flores)","🎗️ Breast Cancer (tumores)","🍷 Wine (vinos)"])

    if multi_ds and st.button("🚀 Lanzar benchmark multi-dataset", type="primary"):
        quick_models = {
            "DT depth=5":        DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Grad. Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        quick_rows = []
        with st.spinner("Calculando…"):
            for ds_name in multi_ds:
                X_q, y_q, _, _ = load_dataset(ds_name)
                for mname, m in quick_models.items():
                    sc = cross_val_score(m, X_q, y_q, cv=5, scoring='accuracy', n_jobs=-1)
                    quick_rows.append({'Dataset':ds_name, 'Modelo':mname,
                                        'CV Mean':round(sc.mean(),4), 'CV Std':round(sc.std(),4)})
        df_q = pd.DataFrame(quick_rows)
        fig_q = px.bar(df_q, x='Dataset', y='CV Mean', color='Modelo', barmode='group',
                        error_y='CV Std', title="Accuracy por Dataset y Modelo",
                        color_discrete_sequence=['#1a9fd4','#2ecc71','#e74c3c'])
        fig_q.update_layout(height=400, **PLOTLY_BASE)
        st.plotly_chart(fig_q, use_container_width=True)
        pivot_q = df_q.pivot(index='Dataset', columns='Modelo', values='CV Mean')
        st.dataframe(pivot_q.style.background_gradient(cmap='Blues'), use_container_width=True)


# ═══════════════════════════════════════════════════════
#  10 — LAB
# ═══════════════════════════════════════════════════════

def show_lab():
    st.markdown('<div class="section-title">🔬 Laboratorio Libre</div>', unsafe_allow_html=True)
    st.markdown("""<div class="concept-box">
    Experimenta con cualquier combinación de modelo, dataset y parámetros.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: lab_model = st.selectbox("Modelo", ["Decision Tree","Random Forest","AdaBoost","Gradient Boosting","Bagging"])
    with c2: lab_ds    = st.selectbox("Dataset", list(DATASET_INFO.keys()))
    with c3: lab_ts    = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)

    X_lab, y_lab, fn_lab, cn_lab = load_dataset(lab_ds)
    st.markdown("### ⚙️ Parámetros")
    pc = st.columns(4)
    if lab_model == "Decision Tree":
        with pc[0]: p1 = st.selectbox("criterion",["gini","entropy"])
        with pc[1]: p2 = st.slider("max_depth",1,20,5,key="lp2")
        with pc[2]: p3 = st.slider("min_samples_split",2,50,2,key="lp3")
        with pc[3]: p4 = st.slider("min_samples_leaf",1,30,1,key="lp4")
        model_lab = DecisionTreeClassifier(criterion=p1,max_depth=p2,min_samples_split=p3,min_samples_leaf=p4,random_state=42)
    elif lab_model == "Random Forest":
        with pc[0]: p1 = st.slider("n_estimators",10,300,100,key="lp1")
        with pc[1]: p2 = st.slider("max_depth",1,20,5,key="lp2")
        with pc[2]: p3 = st.selectbox("max_features",["sqrt","log2"])
        with pc[3]: p4 = st.selectbox("class_weight",[None,"balanced"])
        model_lab = RandomForestClassifier(n_estimators=p1,max_depth=p2,max_features=p3,class_weight=p4,random_state=42,n_jobs=-1)
    elif lab_model == "AdaBoost":
        with pc[0]: p1 = st.slider("n_estimators",10,300,100,key="lp1")
        with pc[1]: p2 = st.select_slider("learning_rate",[0.01,0.05,0.1,0.5,1.0],value=1.0,key="lp2")
        model_lab = make_adaboost(p1)
    elif lab_model == "Gradient Boosting":
        with pc[0]: p1 = st.slider("n_estimators",10,300,100,key="lp1")
        with pc[1]: p2 = st.select_slider("learning_rate",[0.01,0.05,0.1,0.2,0.5],value=0.1,key="lp2")
        with pc[2]: p3 = st.slider("max_depth",1,10,3,key="lp3")
        with pc[3]: p4 = st.slider("subsample",0.3,1.0,1.0,0.1,key="lp4")
        model_lab = GradientBoostingClassifier(n_estimators=p1,learning_rate=p2,max_depth=p3,subsample=p4,random_state=42)
    else:
        with pc[0]: p1 = st.slider("n_estimators",5,200,50,key="lp1")
        with pc[1]: p2 = st.slider("max_samples",0.3,1.0,1.0,0.1,key="lp2")
        with pc[2]: p3 = st.slider("max_features (ratio)",0.3,1.0,1.0,0.1,key="lp3")
        model_lab = BaggingClassifier(n_estimators=p1,max_samples=p2,max_features=p3,random_state=42,n_jobs=-1)

    if st.button("🚀 Entrenar y Evaluar", type="primary"):
        with st.spinner("Entrenando…"):
            X_tr, X_te, y_tr, y_te = train_test_split(X_lab, y_lab, test_size=lab_ts, random_state=42, stratify=y_lab)
            model_lab.fit(X_tr, y_tr); yp = model_lab.predict(X_te)
            cv_sc = cross_val_score(model_lab, X_lab, y_lab, cv=5, scoring='accuracy', n_jobs=-1)

        tr_a = model_lab.score(X_tr,y_tr); te_a = accuracy_score(y_te,yp)
        cr = st.columns(5)
        cr[0].metric("Train Acc",  f"{tr_a:.4f}")
        cr[1].metric("Test Acc",   f"{te_a:.4f}")
        cr[2].metric("F1 (macro)", f"{f1_score(y_te,yp,average='macro'):.4f}")
        cr[3].metric("CV Mean",    f"{cv_sc.mean():.4f}")
        cr[4].metric("CV Std",     f"{cv_sc.std():.4f}")

        fig_l, ax_l = plt.subplots(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_te,yp), annot=True, fmt='d', cmap='Blues', ax=ax_l,
                    xticklabels=[str(c) for c in cn_lab], yticklabels=[str(c) for c in cn_lab])
        ax_l.set_title(f"Matriz de Confusión — {lab_model}", fontweight='bold')
        ax_l.set_xlabel("Predicho"); ax_l.set_ylabel("Real")
        st.pyplot(fig_l); plt.close()

        st.code(classification_report(y_te, yp, target_names=[str(c) for c in cn_lab]), language="text")

        gap_l = tr_a - te_a
        if gap_l > 0.1:
            st.markdown(f'<div class="warning-box">⚠️ Gap={gap_l:.3f} → OVERFITTING. Reduce complejidad.</div>', unsafe_allow_html=True)
        elif te_a < 0.70:
            st.markdown(f'<div class="warning-box">🔵 Test={te_a:.3f} → posible UNDERFITTING.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ Buen ajuste (gap={gap_l:.3f}, Test={te_a:.3f}).</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════

if   module == "home":           show_home()
elif module == "bias_variance":  show_bias_variance()
elif module == "decision_trees": show_decision_trees()
elif module == "entropy":        show_entropy()
elif module == "metrics":        show_metrics()
elif module == "validation":     show_validation()
elif module == "imbalanced":     show_imbalanced()
elif module == "bagging":        show_bagging()
elif module == "boosting":       show_boosting()
elif module == "comparison":     show_comparison()
elif module == "lab":            show_lab()
