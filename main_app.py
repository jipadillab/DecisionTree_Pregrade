"""
🌳 Plataforma Educativa: Árboles de Decisión & Ensambles
Universidad EAFIT — Artificial Intelligence Course
Premium UI Edition
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

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ML Trees & Ensembles · EAFIT",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
#  DESIGN SYSTEM — full CSS overhaul
#  Aesthetic: refined dark-forest · deep teals, amber accents,
#  clean mono typography, glassy cards, subtle grain texture
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

/* ── Root tokens ── */
:root {
    --c-bg:        #0e1117;
    --c-surface:   #161b27;
    --c-surface2:  #1e2636;
    --c-border:    rgba(255,255,255,0.07);
    --c-teal:      #2dd4bf;
    --c-teal-dim:  #1a9fa0;
    --c-amber:     #f59e0b;
    --c-amber-dim: #d97706;
    --c-rose:      #fb7185;
    --c-green:     #34d399;
    --c-text:      #e2e8f0;
    --c-muted:     #94a3b8;
    --c-code-bg:   #0d1117;
    --r-sm: 8px;
    --r-md: 12px;
    --r-lg: 18px;
    --shadow-card: 0 4px 24px rgba(0,0,0,0.4);
    --shadow-glow: 0 0 32px rgba(45,212,191,0.12);
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--c-text) !important;
}
.stApp { background: var(--c-bg) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ══════════ SIDEBAR ══════════ */
[data-testid="stSidebar"] {
    background: var(--c-surface) !important;
    border-right: 1px solid var(--c-border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}

/* sidebar brand */
.sb-brand {
    padding: 1.5rem 1.4rem 1rem;
    border-bottom: 1px solid var(--c-border);
    margin-bottom: 0.5rem;
}
.sb-brand .logo {
    font-size: 1.8rem;
    line-height: 1;
}
.sb-brand h2 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--c-teal) !important;
    margin: 0.4rem 0 0.1rem;
    letter-spacing: -0.01em;
}
.sb-brand p {
    font-size: 0.72rem;
    color: var(--c-muted) !important;
    margin: 0;
    font-variant: small-caps;
    letter-spacing: 0.08em;
}

/* nav section label */
.sb-section-label {
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--c-muted) !important;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.9rem 1.4rem 0.3rem;
}

/* nav pill button */
.nav-btn {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 1.4rem;
    margin: 0.1rem 0.6rem;
    border-radius: var(--r-sm);
    cursor: pointer;
    transition: all 0.18s ease;
    text-decoration: none !important;
    border: none;
    background: transparent;
    width: calc(100% - 1.2rem);
    text-align: left;
}
.nav-btn:hover {
    background: rgba(45,212,191,0.08) !important;
}
.nav-btn.active {
    background: rgba(45,212,191,0.15) !important;
    box-shadow: inset 3px 0 0 var(--c-teal);
}
.nav-btn .nb-icon {
    font-size: 1rem;
    width: 1.4rem;
    text-align: center;
    flex-shrink: 0;
}
.nav-btn .nb-text {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--c-text) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.nav-btn.active .nb-text {
    color: var(--c-teal) !important;
    font-weight: 600;
}

/* progress bar */
.nav-progress {
    padding: 0.8rem 1.4rem 0;
    border-top: 1px solid var(--c-border);
    margin-top: 0.5rem;
}
.nav-progress .prog-label {
    font-size: 0.65rem;
    color: var(--c-muted) !important;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
}
.prog-track {
    height: 3px;
    background: var(--c-surface2);
    border-radius: 99px;
    overflow: hidden;
}
.prog-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--c-teal), var(--c-amber));
    border-radius: 99px;
    transition: width 0.4s ease;
}

/* glossary */
.sb-glossary {
    padding: 0.8rem 1.4rem;
    border-top: 1px solid var(--c-border);
    margin-top: 0.5rem;
}
.sb-glossary summary {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--c-muted) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: pointer;
    padding: 0.4rem 0;
    list-style: none;
}
.sb-glossary summary::marker { display: none; }
.sb-glossary summary::before { content: "▸ "; color: var(--c-teal); }
.sb-glossary[open] summary::before { content: "▾ "; }
.sb-glossary .gl-term {
    display: flex;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.75rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sb-glossary .gl-key {
    font-weight: 600;
    color: var(--c-teal) !important;
    font-family: var(--font-mono);
    flex-shrink: 0;
    width: 7rem;
}
.sb-glossary .gl-val { color: var(--c-muted) !important; }

/* hide default radio widget */
[data-testid="stRadio"] { display: none !important; }

/* ══════════ PAGE HEADER ══════════ */
.page-hero {
    position: relative;
    background: linear-gradient(135deg, #0d2f3a 0%, #0e1c2e 60%, #1a0e2e 100%);
    border: 1px solid var(--c-border);
    border-radius: var(--r-lg);
    padding: 2rem 2.5rem;
    margin-bottom: 1.8rem;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -40%;
    right: -10%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(45,212,191,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.page-hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 20%;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(245,158,11,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-hero .hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(45,212,191,0.12);
    border: 1px solid rgba(45,212,191,0.3);
    border-radius: 99px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--c-teal) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.page-hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #fff !important;
    margin: 0 0 0.4rem;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.page-hero p {
    font-size: 0.9rem;
    color: var(--c-muted) !important;
    margin: 0;
    max-width: 520px;
    line-height: 1.6;
}

/* ══════════ SECTION HEADER ══════════ */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0 1.4rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--c-border);
}
.section-header .sh-icon {
    width: 2.4rem;
    height: 2.4rem;
    border-radius: var(--r-sm);
    background: rgba(45,212,191,0.12);
    border: 1px solid rgba(45,212,191,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.section-header h2 {
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff !important;
    margin: 0 0 0.15rem;
    letter-spacing: -0.02em;
}
.section-header p {
    font-size: 0.8rem;
    color: var(--c-muted) !important;
    margin: 0;
}

/* ══════════ CONTENT CARDS ══════════ */
.card {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--r-md);
    padding: 1.2rem 1.4rem;
    margin: 0.7rem 0;
    position: relative;
}
.card-teal  { border-left: 3px solid var(--c-teal);  background: rgba(45,212,191,0.04); }
.card-amber { border-left: 3px solid var(--c-amber); background: rgba(245,158,11,0.04); }
.card-rose  { border-left: 3px solid var(--c-rose);  background: rgba(251,113,133,0.04); }
.card-green { border-left: 3px solid var(--c-green); background: rgba(52,211,153,0.04); }

.card h4 {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--c-teal) !important;
    margin: 0 0 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.card-amber h4 { color: var(--c-amber) !important; }
.card-rose  h4 { color: var(--c-rose)  !important; }
.card-green h4 { color: var(--c-green) !important; }

.card p, .card li { 
    font-size: 0.84rem; 
    line-height: 1.65; 
    color: var(--c-text) !important;
    margin: 0.15rem 0;
}
.card code {
    font-family: var(--font-mono) !important;
    background: var(--c-code-bg) !important;
    color: var(--c-teal) !important;
    padding: 0.1em 0.4em;
    border-radius: 4px;
    font-size: 0.82em;
}

/* formula card */
.formula-card {
    background: var(--c-code-bg);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: var(--r-md);
    padding: 1.1rem 1.4rem;
    margin: 0.7rem 0;
    font-family: var(--font-mono);
    font-size: 0.85rem;
    line-height: 1.8;
    color: var(--c-text) !important;
    position: relative;
}
.formula-card::before {
    content: 'FÓRMULA';
    position: absolute;
    top: -0.6rem;
    left: 1rem;
    background: var(--c-amber);
    color: #000 !important;
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    padding: 0.1rem 0.5rem;
    border-radius: 99px;
    font-family: var(--font-body);
}
.formula-card b { color: var(--c-amber) !important; }

/* ══════════ STAT BADGES ══════════ */
.stat-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin: 0.8rem 0;
}
.stat-badge {
    flex: 1;
    min-width: 100px;
    background: var(--c-surface2);
    border: 1px solid var(--c-border);
    border-radius: var(--r-sm);
    padding: 0.75rem 1rem;
    text-align: center;
}
.stat-badge .sb-val {
    font-size: 1.4rem;
    font-weight: 700;
    font-family: var(--font-mono);
    color: var(--c-teal) !important;
    display: block;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-badge .sb-label {
    font-size: 0.68rem;
    color: var(--c-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ══════════ ALERT BOXES ══════════ */
.alert {
    display: flex;
    align-items: flex-start;
    gap: 0.7rem;
    padding: 0.85rem 1.1rem;
    border-radius: var(--r-sm);
    margin: 0.7rem 0;
    font-size: 0.84rem;
    line-height: 1.55;
}
.alert .al-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
.alert-warn  { background: rgba(245,158,11,0.1);  border: 1px solid rgba(245,158,11,0.3);  color: #fcd34d !important; }
.alert-ok    { background: rgba(52,211,153,0.1);  border: 1px solid rgba(52,211,153,0.3);  color: #6ee7b7 !important; }
.alert-info  { background: rgba(45,212,191,0.1);  border: 1px solid rgba(45,212,191,0.3);  color: #99f6e4 !important; }
.alert-error { background: rgba(251,113,133,0.1); border: 1px solid rgba(251,113,133,0.3); color: #fda4af !important; }
.alert b { font-weight: 700; }

/* ══════════ TABS ══════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--c-surface) !important;
    border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
    border-bottom: 1px solid var(--c-border) !important;
    gap: 0 !important;
    padding: 0 0.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--c-muted) !important;
    padding: 0.75rem 1.1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    letter-spacing: 0.01em;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--c-text) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--c-teal) !important;
    border-bottom-color: var(--c-teal) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--r-sm) var(--r-sm) !important;
    padding: 1.4rem !important;
}

/* ══════════ STREAMLIT WIDGETS ══════════ */
/* sliders */
[data-testid="stSlider"] > div > div > div { 
    background: var(--c-teal) !important; 
}
[data-testid="stSlider"] > div > div > div > div {
    background: var(--c-teal) !important;
    box-shadow: 0 0 8px rgba(45,212,191,0.5) !important;
}

/* selectbox */
[data-testid="stSelectbox"] > div > div {
    background: var(--c-surface2) !important;
    border-color: var(--c-border) !important;
    border-radius: var(--r-sm) !important;
}

/* number input */
[data-testid="stNumberInput"] input {
    background: var(--c-surface2) !important;
    border-color: var(--c-border) !important;
    border-radius: var(--r-sm) !important;
    font-family: var(--font-mono) !important;
    color: var(--c-text) !important;
}

/* checkbox */
[data-testid="stCheckbox"] label { font-size: 0.84rem !important; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--c-teal-dim), var(--c-teal)) !important;
    color: #0e1117 !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.4rem !important;
    letter-spacing: 0.02em;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 12px rgba(45,212,191,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(45,212,191,0.4) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--c-amber-dim), var(--c-amber)) !important;
    box-shadow: 0 2px 12px rgba(245,158,11,0.3) !important;
}

/* metrics */
[data-testid="stMetric"] {
    background: var(--c-surface2) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-md) !important;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--c-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    color: var(--c-teal) !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* code blocks */
[data-testid="stCodeBlock"] {
    background: var(--c-code-bg) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--r-sm) !important;
}
.stCode, pre { font-family: var(--font-mono) !important; }

/* dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important;
    overflow: hidden;
}

/* expander */
[data-testid="stExpander"] {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: var(--r-sm) !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: var(--c-text) !important;
}

/* spinner */
[data-testid="stSpinner"] { color: var(--c-teal) !important; }

/* multiselect */
[data-baseweb="tag"] {
    background: rgba(45,212,191,0.15) !important;
    border-color: rgba(45,212,191,0.3) !important;
}

/* hr */
hr { border-color: var(--c-border) !important; margin: 1.5rem 0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--c-bg); }
::-webkit-scrollbar-thumb { background: var(--c-surface2); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--c-teal-dim); }

/* ══════════ HOME GRID ══════════ */
.module-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.75rem;
    margin: 1rem 0;
}
.module-tile {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--r-md);
    padding: 1rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    transition: all 0.2s ease;
    cursor: default;
}
.module-tile:hover {
    border-color: rgba(45,212,191,0.35);
    background: rgba(45,212,191,0.04);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}
.module-tile .mt-icon {
    font-size: 1.5rem;
    width: 2.5rem;
    height: 2.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: var(--c-surface2);
    border-radius: var(--r-sm);
    flex-shrink: 0;
}
.module-tile .mt-body h4 {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--c-text) !important;
    margin: 0 0 0.2rem;
}
.module-tile .mt-body p {
    font-size: 0.76rem;
    color: var(--c-muted) !important;
    margin: 0;
    line-height: 1.4;
}

.dataset-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin: 0.8rem 0;
}
.ds-chip {
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: var(--r-sm);
    padding: 0.75rem 1rem;
}
.ds-chip .dc-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--c-text) !important;
    margin-bottom: 0.2rem;
}
.ds-chip .dc-meta {
    font-size: 0.7rem;
    color: var(--c-muted) !important;
    line-height: 1.4;
}
.tag-binary { color: #60a5fa !important; font-size: 0.65rem; font-weight: 700;
    background: rgba(96,165,250,0.1); padding: 0.1rem 0.45rem; border-radius: 99px;
    letter-spacing: 0.05em; text-transform: uppercase; margin-left: 0.4rem; }
.tag-multi  { color: #fb923c !important; font-size: 0.65rem; font-weight: 700;
    background: rgba(251,146,60,0.1); padding: 0.1rem 0.45rem; border-radius: 99px;
    letter-spacing: 0.05em; text-transform: uppercase; margin-left: 0.4rem; }

/* diagnosis badge */
.diagnosis {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: var(--r-sm);
    font-size: 0.85rem;
    font-weight: 700;
    font-family: var(--font-mono);
    letter-spacing: 0.02em;
    width: 100%;
    justify-content: center;
}
.dx-over  { background: rgba(251,113,133,0.15); border: 1px solid rgba(251,113,133,0.35); color: #fda4af !important; }
.dx-under { background: rgba(96,165,250,0.15);  border: 1px solid rgba(96,165,250,0.35);  color: #93c5fd !important; }
.dx-good  { background: rgba(52,211,153,0.15);  border: 1px solid rgba(52,211,153,0.35);  color: #6ee7b7 !important; }

/* step list */
.step-list { counter-reset: step; padding: 0; list-style: none; margin: 0; }
.step-list li {
    counter-increment: step;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.84rem;
    color: var(--c-text) !important;
}
.step-list li::before {
    content: counter(step);
    min-width: 1.4rem;
    height: 1.4rem;
    background: rgba(45,212,191,0.15);
    border: 1px solid rgba(45,212,191,0.3);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--c-teal) !important;
    flex-shrink: 0;
    font-family: var(--font-mono);
    margin-top: 0.05rem;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MATPLOTLIB — transparent, theme-adaptive
# ══════════════════════════════════════════════════════════
plt.rcParams.update({
    "figure.facecolor":  "none",
    "axes.facecolor":    "#161b27",
    "savefig.facecolor": "none",
    "axes.edgecolor":    "#2d3748",
    "grid.color":        "#2d3748",
    "grid.alpha":        0.4,
    "text.color":        "#e2e8f0",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
})

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,27,39,1)',
    font=dict(family='DM Sans, sans-serif', color='#e2e8f0', size=12),
    xaxis=dict(gridcolor='#2d3748', linecolor='#2d3748', zerolinecolor='#2d3748'),
    yaxis=dict(gridcolor='#2d3748', linecolor='#2d3748', zerolinecolor='#2d3748'),
)

# ══════════════════════════════════════════════════════════
#  MODULE REGISTRY
# ══════════════════════════════════════════════════════════
MODULES = [
    ("home",          "🏠", "Inicio",                    "Bienvenida y mapa del curso"),
    ("bias_variance", "📊", "Underfitting & Overfitting", "Sesgo, varianza y trade-off"),
    ("decision_trees","🌲", "Árboles de Decisión",        "Estructura, splits y poda"),
    ("entropy",       "📐", "Entropía & Info Gain",       "Cómo el árbol elige el split"),
    ("metrics",       "📏", "Métricas de Desempeño",      "Accuracy, F1, ROC-AUC"),
    ("validation",    "✅", "Validación & CV",             "Hold-out, K-Fold, curvas"),
    ("imbalanced",    "⚖️", "Clases Desbalanceadas",       "SMOTE, undersampling, weights"),
    ("bagging",       "🌳", "Bagging & Random Forest",    "Ensamble paralelo, OOB"),
    ("boosting",      "🚀", "Boosting",                   "AdaBoost, Gradient Boosting"),
    ("comparison",    "🏆", "Comparación de Modelos",     "Benchmark multi-dataset"),
    ("lab",           "🔬", "Laboratorio Libre",          "Experimenta libremente"),
]
MODULE_IDS   = [m[0] for m in MODULES]
MODULE_TOTAL = len(MODULES) - 1  # exclude home

GLOSSARY = [
    ("Underfitting",   "alto sesgo, modelo muy simple"),
    ("Overfitting",    "alta varianza, memoriza train"),
    ("Entropía",       "medida de impureza/desorden"),
    ("Gini",           "impureza alternativa a entropía"),
    ("Bagging",        "modelos en paralelo → ↓ varianza"),
    ("Boosting",       "secuencial → corrige errores → ↓ sesgo"),
    ("Random Forest",  "bagging + feature subsampling"),
    ("SMOTE",          "oversampling sintético por interpolación"),
    ("OOB Score",      "validación gratuita out-of-bag"),
    ("CV",             "cross-validation k-fold"),
]

# ══════════════════════════════════════════════════════════
#  SESSION STATE — navigation
# ══════════════════════════════════════════════════════════
if "module" not in st.session_state:
    st.session_state.module = "home"

def nav_to(mod_id):
    st.session_state.module = mod_id

# ══════════════════════════════════════════════════════════
#  SIDEBAR — custom nav
# ══════════════════════════════════════════════════════════
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sb-brand">
        <div class="logo">🌳</div>
        <h2>Trees & Ensembles</h2>
        <p>Universidad EAFIT · AI Course</p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-section-label">Módulos</div>', unsafe_allow_html=True)

    for mod_id, icon, label, _ in MODULES:
        is_active = st.session_state.module == mod_id
        active_cls = "active" if is_active else ""
        # Use a button per module
        if st.button(
            f"{icon}  {label}",
            key=f"nav_{mod_id}",
            use_container_width=True,
            type="secondary",
        ):
            nav_to(mod_id)

    # Progress bar
    current_idx = MODULE_IDS.index(st.session_state.module)
    pct = int(current_idx / MODULE_TOTAL * 100) if current_idx > 0 else 0
    st.markdown(f"""
    <div class="nav-progress">
        <div class="prog-label">
            <span>Progreso</span>
            <span>{current_idx}/{MODULE_TOTAL}</span>
        </div>
        <div class="prog-track"><div class="prog-fill" style="width:{pct}%"></div></div>
    </div>
    """, unsafe_allow_html=True)

    # Glossary
    st.markdown("""
    <details class="sb-glossary">
        <summary>Glosario rápido</summary>
        <div style="margin-top:0.5rem">
    """ + "".join([
        f'<div class="gl-term"><span class="gl-key">{k}</span><span class="gl-val">{v}</span></div>'
        for k, v in GLOSSARY
    ]) + """
        </div>
    </details>
    """, unsafe_allow_html=True)

# Re-style nav buttons after render
st.markdown("""
<style>
/* Override Streamlit button style inside sidebar only */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.52rem 1rem !important;
    border-radius: 7px !important;
    transition: all 0.15s ease !important;
    border: none !important;
    letter-spacing: 0 !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(45,212,191,0.08) !important;
    color: #e2e8f0 !important;
    transform: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

module = st.session_state.module

# ══════════════════════════════════════════════════════════
#  DATASETS
# ══════════════════════════════════════════════════════════
DATASET_INFO = {
    "🌸 Iris (flores)":           {"desc": "3 especies · 150 muestras · 4 features",        "binary": False},
    "🍷 Wine (vinos)":            {"desc": "3 tipos de vino · 178 muestras · 13 features",  "binary": False},
    "🎗️ Breast Cancer":          {"desc": "Maligno vs benigno · 569 muestras · 30 feat.",  "binary": True},
    "🌙 Moons (2D)":              {"desc": "Dos lunas · 2 features · ideal para fronteras", "binary": True},
    "⭕ Circles (2D)":            {"desc": "Círculos concéntricos · 2 features",            "binary": True},
    "🎲 Sintético balanceado":    {"desc": "10 features · clases bien separadas",           "binary": True},
    "⚠️ Sintético desbalanceado": {"desc": "85 % cls 0 · 15 % cls 1 · simula fraude",      "binary": True},
}

def load_dataset(name, n_samples=500, noise=0.2, seed=42):
    np.random.seed(seed)
    if name == "🌸 Iris (flores)":
        d = load_iris()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🍷 Wine (vinos)":
        d = load_wine()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🎗️ Breast Cancer":
        d = load_breast_cancer()
        return d.data, d.target, list(d.feature_names), list(d.target_names)
    elif name == "🌙 Moons (2D)":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    elif name == "⭕ Circles (2D)":
        X, y = make_circles(n_samples=n_samples, noise=noise*0.6, factor=0.5, random_state=seed)
        return X, y, ["Feature 1", "Feature 2"], ["Clase 0", "Clase 1"]
    elif name == "🎲 Sintético balanceado":
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5, random_state=seed)
        return X, y, [f"Feat_{i}" for i in range(10)], ["Clase 0", "Clase 1"]
    else:
        X, y = make_classification(n_samples=n_samples, n_features=10, n_informative=5,
                                    weights=[0.85,0.15], random_state=seed)
        return X, y, [f"Feat_{i}" for i in range(10)], ["Cls 0 (may.)", "Cls 1 (min.)"]

def make_adaboost(n_estimators=100):
    try:
        return AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    except TypeError:
        return AdaBoostClassifier(n_estimators=n_estimators, random_state=42, algorithm='SAMME')

def plot_decision_boundary(model, X, y, title="", ax=None):
    h = 0.05
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h), np.arange(y_min,y_max,h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    if ax is None: fig, ax = plt.subplots(figsize=(6,4))
    else: fig = ax.get_figure()
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
    scatter_colors = ['#2dd4bf' if c==0 else '#fb7185' for c in y]
    ax.scatter(X[:,0], X[:,1], c=scatter_colors, edgecolors='rgba(0,0,0,0.5)',
               linewidths=0.5, s=35, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Feature 1", fontsize=9); ax.set_ylabel("Feature 2", fontsize=9)
    return fig

def entropy_fn(p):
    if p<=0 or p>=1: return 0.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def gini_fn(p): return 1 - p**2 - (1-p)**2

def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <div class="sh-icon">{icon}</div>
        <div>
            <h2>{title}</h2>
            {'<p>'+subtitle+'</p>' if subtitle else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

def card(content, variant="teal"):
    st.markdown(f'<div class="card card-{variant}">{content}</div>', unsafe_allow_html=True)

def formula_card(content):
    st.markdown(f'<div class="formula-card">{content}</div>', unsafe_allow_html=True)

def alert(content, kind="info"):
    icons = {"info":"💡","warn":"⚠️","ok":"✅","error":"🔴"}
    st.markdown(f'<div class="alert alert-{kind}"><span class="al-icon">{icons[kind]}</span><div>{content}</div></div>',
                unsafe_allow_html=True)

def diagnosis_badge(gap, te_acc):
    if gap > 0.1:
        st.markdown('<div class="diagnosis dx-over">🔴 OVERFITTING — gap demasiado alto</div>', unsafe_allow_html=True)
    elif te_acc < 0.65:
        st.markdown('<div class="diagnosis dx-under">🔵 UNDERFITTING — accuracy muy bajo</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="diagnosis dx-good">✅ BUEN AJUSTE</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
#  MODULE: HOME
# ══════════════════════════════════════════════════════════
def show_home():
    st.markdown("""
    <div class="page-hero">
        <div class="hero-tag">🎓 Universidad EAFIT · IA Course</div>
        <h1>Árboles de Decisión<br>& Ensambles</h1>
        <p>Plataforma interactiva para aprender ML supervisado.
        Visualiza, experimenta y comprende cada concepto en tiempo real.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Módulos disponibles")
    topics = [
        ("📊","Underfitting & Overfitting","Sesgo, varianza y el trade-off fundamental"),
        ("🌲","Árboles de Decisión","Estructura, splits, profundidad y poda"),
        ("📐","Entropía & Info Gain","Cómo el árbol elige la mejor pregunta"),
        ("📏","Métricas de Desempeño","Accuracy, Precision, Recall, F1, ROC-AUC"),
        ("✅","Validación & CV","Hold-out, K-Fold, Stratified, curvas"),
        ("⚖️","Clases Desbalanceadas","SMOTE, undersampling, class_weight"),
        ("🌳","Bagging / Random Forest","Ensamble paralelo, OOB score, importancia"),
        ("🚀","Boosting","AdaBoost, Gradient Boosting, staged scores"),
        ("🏆","Comparación de Modelos","Benchmark interactivo multi-dataset"),
        ("🔬","Laboratorio Libre","Experimenta con tus propios parámetros"),
    ]
    html_tiles = '<div class="module-grid">'
    for icon, title, desc in topics:
        html_tiles += f"""
        <div class="module-tile">
            <div class="mt-icon">{icon}</div>
            <div class="mt-body">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
        </div>"""
    html_tiles += '</div>'
    st.markdown(html_tiles, unsafe_allow_html=True)

    st.markdown("<br>#### Datasets incluidos", unsafe_allow_html=True)
    html_ds = '<div class="dataset-grid">'
    for name, info in DATASET_INFO.items():
        tag_cls = "tag-binary" if info["binary"] else "tag-multi"
        tag_lbl = "Binario" if info["binary"] else "Multi-clase"
        html_ds += f"""
        <div class="ds-chip">
            <div class="dc-name">{name}<span class="{tag_cls}">{tag_lbl}</span></div>
            <div class="dc-meta">{info['desc']}</div>
        </div>"""
    html_ds += '</div>'
    st.markdown(html_ds, unsafe_allow_html=True)

    alert("Usa el menú lateral para navegar entre módulos. Cada sección tiene <b>teoría</b>, <b>interactividad</b> y <b>diagnósticos automáticos</b>.", "info")

# ══════════════════════════════════════════════════════════
#  MODULE 1: BIAS-VARIANCE
# ══════════════════════════════════════════════════════════
def show_bias_variance():
    section_header("📊","Underfitting, Overfitting & Bias-Variance","Trade-off entre sesgo y varianza")
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🎮 Árbol & Complejidad", "📉 Curvas de Aprendizaje"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card("""<h4>🔵 Underfitting</h4>
            <p>Modelo <b>demasiado simple</b>.</p>
            <ul style="margin:0.4rem 0 0;padding-left:1.1rem">
            <li>Alto sesgo (bias), baja varianza</li>
            <li>Error alto en train <b>y</b> en test</li>
            <li>Causa: árbol muy shallow, pocas features</li>
            </ul>""", "teal")
        with c2:
            card("""<h4 style="color:var(--c-rose)!important">🔴 Overfitting</h4>
            <p>Modelo que <b>memoriza</b> el entrenamiento.</p>
            <ul style="margin:0.4rem 0 0;padding-left:1.1rem">
            <li>Bajo sesgo, alta varianza</li>
            <li>Error bajo en train, <b>alto</b> en test</li>
            <li>Causa: árbol muy profundo, sin regularización</li>
            </ul>""", "rose")

        formula_card("""<b>Error Total = Sesgo² + Varianza + Ruido irreducible</b>
Reducir uno generalmente aumenta el otro → <i>tradeoff inevitable</i>""")

        complexity = np.linspace(1, 10, 200)
        bias2    = 5 * np.exp(-0.4 * complexity)
        variance = 0.05 * np.exp(0.5 * complexity)
        total    = bias2 + variance + 0.5
        opt_idx  = int(np.argmin(total))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(complexity, bias2,    color='#60a5fa', lw=2.5, label='Sesgo²')
        ax.plot(complexity, variance, color='#fb7185', lw=2.5, label='Varianza')
        ax.plot(complexity, total,    color='#94a3b8', lw=2.5, ls='--', label='Error Total')
        ax.axhline(0.5, color='#475569', ls=':', alpha=0.7, label='Ruido irreducible')
        ax.axvline(complexity[opt_idx], color='#2dd4bf', ls='--', lw=1.5, label='Óptimo')
        ax.fill_between(complexity[:opt_idx+1], 0, total[:opt_idx+1], alpha=0.06, color='#60a5fa')
        ax.fill_between(complexity[opt_idx:],   0, total[opt_idx:],   alpha=0.06, color='#fb7185')
        ax.set_xlabel('Complejidad del Modelo'); ax.set_ylabel('Error')
        ax.set_title('Tradeoff Sesgo-Varianza'); ax.legend(fontsize=9); ax.set_ylim(0, 6)
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            st.markdown("**Configuración**")
            ds_bv    = st.selectbox("Dataset", ["🌙 Moons (2D)","⭕ Circles (2D)","🌸 Iris (flores)"], key="bv_ds")
            noise_bv = st.slider("Ruido", 0.0, 0.5, 0.2, 0.05, key="bv_n2")
            n_bv     = st.slider("Muestras", 100, 600, 300, 50, key="bv_ns")
            depth_bv = st.slider("max_depth", 1, 15, 3, key="bv_d")
            ts_bv    = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, key="bv_ts")

        X_bv, y_bv, _, _ = load_dataset(ds_bv, n_samples=n_bv, noise=noise_bv)
        X_bv = X_bv[:, :2]
        X_tr, X_te, y_tr, y_te = train_test_split(X_bv, y_bv, test_size=ts_bv, random_state=42, stratify=y_bv)
        m_bv = DecisionTreeClassifier(max_depth=depth_bv, random_state=42).fit(X_tr, y_tr)
        tr_acc = m_bv.score(X_tr, y_tr); te_acc = m_bv.score(X_te, y_te); gap = tr_acc - te_acc

        with c2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_decision_boundary(m_bv, X_tr, y_tr, f"Train | Acc={tr_acc:.2%}", axes[0])
            plot_decision_boundary(m_bv, X_te, y_te, f"Test  | Acc={te_acc:.2%}", axes[1])
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        mc = st.columns(3)
        mc[0].metric("Train Acc", f"{tr_acc:.2%}")
        mc[1].metric("Test Acc",  f"{te_acc:.2%}", delta=f"{te_acc-tr_acc:+.2%}")
        mc[2].metric("Gap",       f"{gap:.2%}", delta="overfitting" if gap>0.1 else "ok")
        diagnosis_badge(gap, te_acc)

        depths_sw = list(range(1, 16))
        tr_sw = [DecisionTreeClassifier(max_depth=d,random_state=42).fit(X_tr,y_tr).score(X_tr,y_tr) for d in depths_sw]
        te_sw = [DecisionTreeClassifier(max_depth=d,random_state=42).fit(X_tr,y_tr).score(X_te,y_te) for d in depths_sw]
        fig2 = go.Figure([
            go.Scatter(x=depths_sw, y=tr_sw, mode='lines+markers', name='Train',
                       line=dict(color='#fb7185',width=2.5), marker=dict(size=6)),
            go.Scatter(x=depths_sw, y=te_sw, mode='lines+markers', name='Test',
                       line=dict(color='#2dd4bf',width=2.5), marker=dict(size=6)),
        ])
        fig2.add_vline(x=depth_bv, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"depth={depth_bv}", annotation_font_color="#f59e0b")
        fig2.update_layout(xaxis_title="Profundidad", yaxis_title="Accuracy",
                            title="Accuracy vs Profundidad del árbol",
                            height=300, margin=dict(t=40,b=40,l=40,r=20), **PLOTLY_BASE)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            ds_lc   = st.selectbox("Dataset", ["🎗️ Breast Cancer","🌸 Iris (flores)","🍷 Wine (vinos)"], key="lc_ds")
            depth_lc = st.slider("max_depth", 1, 15, 5, key="lc_d")
        X_lc, y_lc, _, _ = load_dataset(ds_lc)
        tr_sz, tr_sc, va_sc = learning_curve(
            DecisionTreeClassifier(max_depth=depth_lc,random_state=42),
            X_lc, y_lc, cv=5, train_sizes=np.linspace(0.1,1.0,10), scoring='accuracy', n_jobs=-1)
        with c2:
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(tr_sz, tr_sc.mean(1), color='#fb7185', lw=2.5, marker='o', ms=5, label='Train')
            ax.fill_between(tr_sz, tr_sc.mean(1)-tr_sc.std(1), tr_sc.mean(1)+tr_sc.std(1), alpha=0.1, color='#fb7185')
            ax.plot(tr_sz, va_sc.mean(1), color='#2dd4bf', lw=2.5, marker='o', ms=5, label='Validación')
            ax.fill_between(tr_sz, va_sc.mean(1)-va_sc.std(1), va_sc.mean(1)+va_sc.std(1), alpha=0.1, color='#2dd4bf')
            ax.set_xlabel("Tamaño conjunto train"); ax.set_ylabel("Accuracy")
            ax.set_title(f"Curvas de Aprendizaje — max_depth={depth_lc}"); ax.legend()
            st.pyplot(fig, use_container_width=True); plt.close()
        vf = va_sc.mean(1)[-1]; tf = tr_sc.mean(1)[-1]; gap_lc = tf - vf
        if gap_lc > 0.1:
            alert(f"<b>OVERFITTING</b> — gap={gap_lc:.2%}. Considera reducir max_depth.", "warn")
        elif vf < 0.75:
            alert(f"<b>UNDERFITTING</b> — val={vf:.2%}. Aumenta la complejidad del modelo.", "warn")
        else:
            alert(f"<b>Buen ajuste</b> — val={vf:.2%}, gap={gap_lc:.2%}. Modelo bien calibrado.", "ok")

# ══════════════════════════════════════════════════════════
#  MODULE 2: DECISION TREES
# ══════════════════════════════════════════════════════════
def show_decision_trees():
    section_header("🌲","Árboles de Decisión","Algoritmo CART: estructura, splits y hiperparámetros")
    tab1, tab2, tab3 = st.tabs(["📖 Conceptos", "🎮 Constructor Interactivo", "🔍 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card("""<h4>Componentes del árbol</h4>
            <ul style="margin:0;padding-left:1.1rem">
            <li><b>Nodo raíz</b>: primera pregunta, más informativa</li>
            <li><b>Nodo interno</b>: condición sobre una feature</li>
            <li><b>Hoja (leaf)</b>: predicción final</li>
            <li><b>Profundidad</b>: niveles raíz → hoja más lejana</li>
            </ul>""", "teal")
        with c2:
            card("""<h4>Hiperparámetros clave</h4>
            <ul style="margin:0;padding-left:1.1rem">
            <li><code>max_depth</code> — controla overfitting</li>
            <li><code>min_samples_split</code> — mínimo para hacer un split</li>
            <li><code>min_samples_leaf</code> — mínimo muestras en hojas</li>
            <li><code>criterion</code> — <code>'gini'</code> o <code>'entropy'</code></li>
            <li><code>ccp_alpha</code> — poda (cost-complexity)</li>
            </ul>""", "amber")

        st.markdown("**Algoritmo CART — pasos**")
        st.markdown("""<ol class="step-list">
        <li>Para cada feature y umbral posible, calcular la impureza del split resultante</li>
        <li>Elegir el split que maximiza el <b>Information Gain</b> (min impureza ponderada)</li>
        <li>Dividir el nodo en dos hijos y repetir recursivamente</li>
        <li>Detener cuando se cumple: <code>max_depth</code>, <code>min_samples</code>, o nodo puro</li>
        </ol>""", unsafe_allow_html=True)

    with tab2:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            dt_ds  = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="dt_ds")
            dt_cr  = st.selectbox("criterion", ["gini","entropy"], key="dt_cr")
            dt_md  = st.slider("max_depth", 1, 12, 3, key="dt_md")
            dt_mss = st.slider("min_samples_split", 2, 50, 2, key="dt_mss")
            dt_msl = st.slider("min_samples_leaf",  1, 30, 1, key="dt_msl")
            dt_ts  = st.slider("Test size", 0.1, 0.5, 0.3, 0.05, key="dt_ts")

        X_dt, y_dt, fn_dt, cn_dt = load_dataset(dt_ds)
        X2 = X_dt[:,:2]
        X_tr, X_te, y_tr, y_te = train_test_split(X2, y_dt, test_size=dt_ts, random_state=42, stratify=y_dt)
        dt = DecisionTreeClassifier(criterion=dt_cr, max_depth=dt_md,
                                     min_samples_split=dt_mss, min_samples_leaf=dt_msl, random_state=42)
        dt.fit(X_tr, y_tr); yp = dt.predict(X_te)

        with c2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            plot_decision_boundary(dt, X_tr, y_tr, "Train", axes[0])
            plot_decision_boundary(dt, X_te, y_te, f"Test Acc={accuracy_score(y_te,yp):.2%}", axes[1])
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        mc = st.columns(4)
        mc[0].metric("Accuracy (Test)",  f"{accuracy_score(y_te,yp):.3f}")
        mc[1].metric("Nodos totales",    dt.tree_.node_count)
        mc[2].metric("Profundidad real", dt.get_depth())
        mc[3].metric("Hojas",            dt.get_n_leaves())

        st.markdown("**Estructura visual del árbol**")
        fig_t, ax_t = plt.subplots(figsize=(max(10, dt_md*3.5), max(5, dt_md*2)))
        fig_t.patch.set_facecolor('#0e1117')
        plot_tree(dt, feature_names=fn_dt[:2], class_names=[str(c) for c in cn_dt],
                  filled=True, rounded=True, fontsize=8, ax=ax_t)
        ax_t.set_title(f"criterion={dt_cr}, max_depth={dt_md}", color='#e2e8f0')
        st.pyplot(fig_t, use_container_width=True); plt.close()

    with tab3:
        c1, c2 = st.columns([1, 2])
        with c1:
            fi_ds = st.selectbox("Dataset", ["🎗️ Breast Cancer","🍷 Wine (vinos)","🌸 Iris (flores)"], key="fi_ds")
            fi_d  = st.slider("max_depth", 1, 12, 4, key="fi_d2")
        X_fi, y_fi, fn_fi, _ = load_dataset(fi_ds)
        X_tr_fi, _, y_tr_fi, _ = train_test_split(X_fi, y_fi, test_size=0.3, random_state=42, stratify=y_fi)
        dt_fi = DecisionTreeClassifier(max_depth=fi_d, random_state=42).fit(X_tr_fi, y_tr_fi)
        imp = dt_fi.feature_importances_; sidx = np.argsort(imp)[::-1][:15]

        with c2:
            fig_fi, ax_fi = plt.subplots(figsize=(9, 5))
            colors_fi = ['#2dd4bf' if i==sidx[0] else '#1a9fa0' if i in sidx[:5] else '#164e63' for i in sidx]
            ax_fi.barh(range(len(sidx)), imp[sidx], color=colors_fi, edgecolor='none')
            ax_fi.set_yticks(range(len(sidx)))
            ax_fi.set_yticklabels([fn_fi[i] for i in sidx], fontsize=9)
            ax_fi.set_xlabel("Importancia (Gini Reduction)")
            ax_fi.set_title("Feature Importance")
            plt.tight_layout(); st.pyplot(fig_fi, use_container_width=True); plt.close()

        rules = export_text(dt_fi, feature_names=list(fn_fi), max_depth=3)
        st.code(rules[:2000]+("\n..." if len(rules)>2000 else ""), language="text")

# ══════════════════════════════════════════════════════════
#  MODULE 3: ENTROPY
# ══════════════════════════════════════════════════════════
def show_entropy():
    section_header("📐","Entropía & Information Gain","Cómo el árbol elige la mejor pregunta en cada nodo")
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🧮 Calculadora", "🎮 Gini vs Entropía"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            formula_card("""<b>Entropía de Shannon:</b>
H(π) = −Σ πᵢ · log₂(πᵢ)

Binaria: H = −πₚ·log₂(πₚ) − πₙ·log₂(πₙ)
donde πₚ = p/(p+n),  πₙ = n/(p+n)

H = 0 → nodo puro  |  H = 1 → 50/50 (máximo)""")
        with c2:
            formula_card("""<b>Índice Gini:</b>
Gini = 1 − Σ πᵢ²  =  2p(1−p)  (binario)

<b>Expected Entropy tras split A:</b>
EH(A) = Σᵢ [(pᵢ+nᵢ)/(p+n)] · H(πᵢ)

<b>Information Gain:</b>
IG(A) = H(padre) − EH(A)   → maximizar""")

        p_vals = np.linspace(0.001, 0.999, 200)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(p_vals, [entropy_fn(p) for p in p_vals], color='#60a5fa', lw=2.5, label='Entropía H(p)')
        ax.plot(p_vals, [gini_fn(p)   for p in p_vals], color='#fb7185', lw=2.5, label='Gini 2p(1−p)')
        ax.axvline(0.5, color='#475569', ls='--', alpha=0.6)
        ax.set_xlabel("Proporción clase positiva (p)"); ax.set_ylabel("Impureza")
        ax.set_title("Entropía vs Gini — ambas miden lo mismo con diferente escala"); ax.legend()
        st.pyplot(fig, use_container_width=True); plt.close()

    with tab2:
        st.markdown("**Calcula Information Gain manualmente**")
        c1, c2, c3 = st.columns(3)
        with c1:
            card("<h4>Nodo Padre</h4>", "teal")
            pp = st.number_input("Positivos", 1, 200, 6, key="ig_pp")
            pn = st.number_input("Negativos", 1, 200, 6, key="ig_pn")
        with c2:
            card("<h4>Hijo Izquierdo</h4>", "teal")
            lp = st.number_input("Positivos", 0, 200, 4, key="ig_lp")
            ln = st.number_input("Negativos", 0, 200, 0, key="ig_ln")
        with c3:
            card("<h4>Hijo Derecho</h4>", "teal")
            rp = st.number_input("Positivos", 0, 200, 2, key="ig_rp")
            rn = st.number_input("Negativos", 0, 200, 6, key="ig_rn")

        tot = pp+pn
        h_p = entropy_fn(pp/tot) if tot>0 else 0
        tl, tr2 = lp+ln, rp+rn
        hl = entropy_fn(lp/tl) if tl>0 else 0
        hr = entropy_fn(rp/tr2) if tr2>0 else 0
        eh = (tl/tot)*hl + (tr2/tot)*hr if tot>0 else 0
        ig = h_p - eh

        r1,r2,r3,r4 = st.columns(4)
        r1.metric("H(Padre)", f"{h_p:.4f}"); r2.metric("H(Izq)", f"{hl:.4f}")
        r3.metric("EH(Split)",f"{eh:.4f}");  r4.metric("IG ↑",   f"{ig:.4f}")

        fig_b, ax_b = plt.subplots(figsize=(8,3))
        vals_b = [h_p, hl, hr, eh, ig]
        cols_b = ['#60a5fa','#34d399','#34d399','#fb7185','#f59e0b']
        ax_b.bar(['H(Padre)','H(Izq)','H(Der)','EH(Split)','IG'], vals_b,
                  color=cols_b, edgecolor='none', width=0.6)
        ax_b.set_title(f"Information Gain = {ig:.4f}")
        st.pyplot(fig_b, use_container_width=True); plt.close()

    with tab3:
        gc_ds = st.selectbox("Dataset", ["🌸 Iris (flores)","🎗️ Breast Cancer","🍷 Wine (vinos)"], key="gc_ds")
        gc_d  = st.slider("max_depth", 1, 8, 3, key="gc_d")
        X_gc, y_gc, fn_gc, cn_gc = load_dataset(gc_ds)
        X2 = X_gc[:,:2]
        X_tr, X_te, y_tr, y_te = train_test_split(X2, y_gc, test_size=0.3, random_state=42, stratify=y_gc)
        dtg = DecisionTreeClassifier(criterion='gini',    max_depth=gc_d, random_state=42).fit(X_tr,y_tr)
        dte = DecisionTreeClassifier(criterion='entropy', max_depth=gc_d, random_state=42).fit(X_tr,y_tr)
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        plot_decision_boundary(dtg, X_te, y_te, f"Gini — {dtg.score(X_te,y_te):.2%}", axes[0])
        plot_decision_boundary(dte, X_te, y_te, f"Entropy — {dte.score(X_te,y_te):.2%}", axes[1])
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        alert("En la mayoría de casos <b>Gini y Entropía producen resultados casi idénticos</b>. Gini es más rápido (sin logaritmos). Entropía puede favorecer árboles más balanceados.", "info")

# ══════════════════════════════════════════════════════════
#  MODULE 4: METRICS
# ══════════════════════════════════════════════════════════
def show_metrics():
    section_header("📏","Métricas de Desempeño","Cómo evaluar correctamente un clasificador")
    tab1, tab2, tab3 = st.tabs(["📖 Matriz de Confusión", "📈 ROC & PR Curves", "🧮 Calculadora"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            formula_card("""<b>Métricas desde la matriz de confusión:</b>
Accuracy  = (TP+TN) / Total
Precision = TP / (TP+FP)
Recall    = TP / (TP+FN)
F1-Score  = 2 · P · R / (P+R)
ROC-AUC   = área bajo curva ROC""")
        with c2:
            card("""<h4 style="color:var(--c-amber)!important">⚠️ Trampa de Accuracy</h4>
            <p>Con clases desbalanceadas (95% clase 0, 5% clase 1),
            un modelo que <b>siempre predice clase 0</b> tiene Accuracy=95%
            pero <b>Recall=0%</b> para la clase minoritaria.</p>
            <p><b>Usa F1 o ROC-AUC</b> cuando las clases están desbalanceadas.</p>""", "amber")

        cm_ds = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="cm_ds")
        cm_d  = st.slider("max_depth", 1, 15, 4, key="cm_d")
        X_cm, y_cm, _, cn_cm = load_dataset(cm_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_cm, y_cm, test_size=0.3, random_state=42, stratify=y_cm)
        m_cm = DecisionTreeClassifier(max_depth=cm_d, random_state=42).fit(X_tr, y_tr)
        yp_cm = m_cm.predict(X_te)
        cm_arr = confusion_matrix(y_te, yp_cm)

        c1, c2 = st.columns([1,1.5])
        with c1:
            fig_cm, ax_cm = plt.subplots(figsize=(5,4))
            sns.heatmap(cm_arr, annot=True, fmt='d', cmap='YlOrRd',
                        ax=ax_cm, xticklabels=cn_cm, yticklabels=cn_cm,
                        linewidths=0.5, linecolor='#161b27',
                        cbar_kws={'shrink':0.8})
            ax_cm.set_xlabel("Predicho"); ax_cm.set_ylabel("Real")
            ax_cm.set_title("Matriz de Confusión")
            st.pyplot(fig_cm, use_container_width=True); plt.close()
        with c2:
            n_cls = len(np.unique(y_cm))
            if n_cls == 2:
                mc = st.columns(2)
                mc[0].metric("Accuracy",  f"{accuracy_score(y_te,yp_cm):.3f}")
                mc[1].metric("ROC-AUC",   f"{roc_auc_score(y_te, m_cm.predict_proba(X_te)[:,1]):.3f}")
                mc2 = st.columns(3)
                mc2[0].metric("Precision", f"{precision_score(y_te,yp_cm):.3f}")
                mc2[1].metric("Recall",    f"{recall_score(y_te,yp_cm):.3f}")
                mc2[2].metric("F1",        f"{f1_score(y_te,yp_cm):.3f}")
            st.code(classification_report(y_te, yp_cm, target_names=[str(c) for c in cn_cm]), language="text")

    with tab2:
        roc_ds = st.selectbox("Dataset (binario)", ["🎗️ Breast Cancer","🎲 Sintético balanceado","⚠️ Sintético desbalanceado"], key="roc_ds")
        X_roc, y_roc, _, _ = load_dataset(roc_ds)
        X_tr, X_te, y_tr, y_te = train_test_split(X_roc, y_roc, test_size=0.3, random_state=42, stratify=y_roc)
        fig_roc = make_subplots(rows=1, cols=2, subplot_titles=["Curva ROC","Precision-Recall"])
        for d, lbl, col in zip([2,4,6,None],['depth=2','depth=4','depth=6','full'],
                                 ['#fb7185','#f59e0b','#2dd4bf','#60a5fa']):
            m_ = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr,y_tr)
            proba = m_.predict_proba(X_te)[:,1]
            fpr, tpr, _ = roc_curve(y_te, proba); auc = roc_auc_score(y_te, proba)
            prec, rec, _ = precision_recall_curve(y_te, proba)
            fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{lbl} (AUC={auc:.2f})",line=dict(color=col,width=2)),row=1,col=1)
            fig_roc.add_trace(go.Scatter(x=rec,y=prec,name=lbl,showlegend=False,line=dict(color=col,width=2,dash='dot')),row=1,col=2)
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name='Random',line=dict(dash='dash',color='#475569')),row=1,col=1)
        fig_roc.update_layout(height=420, **PLOTLY_BASE)
        st.plotly_chart(fig_roc, use_container_width=True)

    with tab3:
        st.markdown("**Ingresa TP / FP / FN / TN manualmente**")
        c1, c2 = st.columns(2)
        with c1:
            tp = st.number_input("TP (True Positives)",  0, 1000, 85)
            fp = st.number_input("FP (False Positives)", 0, 1000, 10)
        with c2:
            fn = st.number_input("FN (False Negatives)", 0, 1000, 15)
            tn = st.number_input("TN (True Negatives)",  0, 1000, 90)
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
            fig_c, ax_c = plt.subplots(figsize=(3.5,3))
            sns.heatmap([[tn,fp],[fn,tp]], annot=True, fmt='d', cmap='YlOrRd',
                        ax=ax_c, xticklabels=['Pred−','Pred+'], yticklabels=['Real−','Real+'],
                        linewidths=0.5, linecolor='#161b27')
            st.pyplot(fig_c); plt.close()

# ══════════════════════════════════════════════════════════
#  MODULE 5: VALIDATION
# ══════════════════════════════════════════════════════════
def show_validation():
    section_header("✅","Validación & Cross-Validation","Cómo estimar el rendimiento real de un modelo")
    tab1, tab2, tab3 = st.tabs(["📖 Estrategias", "🎮 K-Fold Interactivo", "🔍 Validation Curve"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card("""<h4>1. Hold-Out</h4><p>Divide en train/test (ej. 70/30).</p>
            <p>✅ Rápido · ❌ Alta varianza en la estimación</p>""","teal")
            card("""<h4>2. K-Fold CV</h4><p>K particiones, K evaluaciones, promedia.</p>
            <p>✅ Estimación robusta · ❌ K× más costoso</p>""","teal")
        with c2:
            card("""<h4>3. Stratified K-Fold</h4><p>Como K-Fold pero mantiene proporción de clases.</p>
            <p>✅ Esencial para datos desbalanceados</p>""","teal")
            card("""<h4>4. Leave-One-Out (LOO)</h4><p>K = N, máximo uso de datos.</p>
            <p>✅ Máximo uso · ❌ Muy costoso para datasets grandes</p>""","teal")

        K = 5
        fig_kf, ax_kf = plt.subplots(figsize=(10, 3.5))
        for fold in range(K):
            for part in range(K):
                col_kf = '#fb7185' if part==fold else '#1a9fa0'
                rect = mpatches.Rectangle([part/K,(K-fold-1)/K+0.05], 1/K-0.01, 0.8/K, color=col_kf, alpha=0.8)
                ax_kf.add_patch(rect)
                lbl = 'Val' if part==fold else 'Tr'
                ax_kf.text(part/K+0.5/K, (K-fold-1)/K+0.05+0.4/K, lbl, ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            ax_kf.text(-0.02, (K-fold-1)/K+0.05+0.4/K, f'Fold {fold+1}', ha='right', va='center', fontsize=9)
        ax_kf.set_xlim(-0.18,1.02); ax_kf.set_ylim(0,1.1); ax_kf.axis('off')
        ax_kf.set_title("K-Fold Cross-Validation (K=5)")
        ax_kf.legend(handles=[mpatches.Patch(color='#1a9fa0',alpha=0.8,label='Train'),
                               mpatches.Patch(color='#fb7185',alpha=0.8,label='Validación')], loc='lower right')
        st.pyplot(fig_kf, use_container_width=True); plt.close()

    with tab2:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            cv_ds = st.selectbox("Dataset", list(DATASET_INFO.keys())[:5], key="cv_ds")
            cv_d  = st.slider("max_depth", 1, 15, 4, key="cv_d")
            k_f   = st.slider("K folds", 3, 20, 5, key="cv_k")
        X_cv, y_cv, _, _ = load_dataset(cv_ds)
        sc_cv = cross_val_score(DecisionTreeClassifier(max_depth=cv_d,random_state=42),
                                 X_cv, y_cv, cv=KFold(k_f, shuffle=True, random_state=42), scoring='accuracy')
        with c2:
            fig_cv, ax_cv = plt.subplots(figsize=(9, 4))
            folds_cv = list(range(1, k_f+1))
            cols_cv = ['#fb7185' if s<sc_cv.mean()-sc_cv.std() else
                        '#2dd4bf' if s>sc_cv.mean()+sc_cv.std() else '#60a5fa' for s in sc_cv]
            bars_cv = ax_cv.bar(folds_cv, sc_cv, color=cols_cv, edgecolor='none', width=0.65)
            ax_cv.axhline(sc_cv.mean(), ls='--', lw=2, color='#f59e0b', label=f'Media={sc_cv.mean():.3f}')
            ax_cv.axhspan(sc_cv.mean()-sc_cv.std(), sc_cv.mean()+sc_cv.std(), alpha=0.08, color='#f59e0b',
                           label=f'±1σ={sc_cv.std():.3f}')
            ax_cv.set_xlabel("Fold"); ax_cv.set_ylabel("Accuracy")
            ax_cv.set_title(f"{k_f}-Fold CV"); ax_cv.legend(fontsize=9)
            ax_cv.set_xticks(folds_cv)
            ax_cv.set_ylim(max(0, sc_cv.min()-0.1), min(1.05, sc_cv.max()+0.05))
            for bar, sc in zip(bars_cv, sc_cv):
                ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004, f'{sc:.3f}',
                            ha='center', va='bottom', fontsize=8, color='#94a3b8')
            st.pyplot(fig_cv, use_container_width=True); plt.close()
        cc = st.columns(4)
        cc[0].metric("Media",  f"{sc_cv.mean():.4f}")
        cc[1].metric("Std",    f"{sc_cv.std():.4f}")
        cc[2].metric("Mínimo", f"{sc_cv.min():.4f}")
        cc[3].metric("Máximo", f"{sc_cv.max():.4f}")

    with tab3:
        vc_ds  = st.selectbox("Dataset", ["🎗️ Breast Cancer","🌸 Iris (flores)","🍷 Wine (vinos)"], key="vc_ds")
        vc_par = st.selectbox("Hiperparámetro", ["max_depth","min_samples_split","min_samples_leaf"], key="vc_p")
        X_vc, y_vc, _, _ = load_dataset(vc_ds)
        pr = np.arange(1,16) if vc_par=='max_depth' else (np.arange(2,50,3) if vc_par=='min_samples_split' else np.arange(1,30,2))
        tr_sc_vc, va_sc_vc = validation_curve(DecisionTreeClassifier(random_state=42), X_vc, y_vc,
                                               param_name=vc_par, param_range=pr, cv=5, scoring='accuracy', n_jobs=-1)
        best_i = int(np.argmax(va_sc_vc.mean(1)))
        fig_vc, ax_vc = plt.subplots(figsize=(9, 4))
        ax_vc.plot(pr, tr_sc_vc.mean(1), color='#fb7185', lw=2.5, marker='o', ms=5, label='Train')
        ax_vc.fill_between(pr, tr_sc_vc.mean(1)-tr_sc_vc.std(1), tr_sc_vc.mean(1)+tr_sc_vc.std(1), alpha=0.1, color='#fb7185')
        ax_vc.plot(pr, va_sc_vc.mean(1), color='#2dd4bf', lw=2.5, marker='o', ms=5, label='Validación')
        ax_vc.fill_between(pr, va_sc_vc.mean(1)-va_sc_vc.std(1), va_sc_vc.mean(1)+va_sc_vc.std(1), alpha=0.1, color='#2dd4bf')
        ax_vc.axvline(pr[best_i], color='#f59e0b', ls='--', lw=1.5, label=f'Mejor: {pr[best_i]}')
        ax_vc.set_xlabel(vc_par); ax_vc.set_ylabel("Accuracy")
        ax_vc.set_title(f"Validation Curve — {vc_par}"); ax_vc.legend()
        st.pyplot(fig_vc, use_container_width=True); plt.close()
        alert(f"Mejor valor: <b>{vc_par} = {pr[best_i]}</b> → CV Accuracy = {va_sc_vc.mean(1)[best_i]:.4f}", "ok")

# ══════════════════════════════════════════════════════════
#  MODULE 6: IMBALANCED
# ══════════════════════════════════════════════════════════
def show_imbalanced():
    section_header("⚖️","Clases Desbalanceadas","Técnicas para corregir distribuciones sesgadas")
    tab1, tab2 = st.tabs(["📖 Técnicas", "🎮 Comparación Interactiva"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            card("""<h4>Oversampling</h4>
            <p>Aumenta la clase minoritaria.</p>
            <ul style="margin:0.4rem 0 0;padding-left:1.1rem">
            <li><b>SMOTE</b>: interpola entre vecinos</li>
            <li><b>ADASYN</b>: SMOTE adaptativo</li>
            </ul>""","teal")
        with c2:
            card("""<h4>Undersampling</h4>
            <p>Reduce la clase mayoritaria.</p>
            <ul style="margin:0.4rem 0 0;padding-left:1.1rem">
            <li><b>Random</b>: elimina aleatoriamente</li>
            <li><b>Tomek Links</b>: elimina pares límite</li>
            </ul>""","amber")
        with c3:
            card("""<h4>class_weight</h4>
            <p>Penaliza errores en clase minoritaria.</p>
            <p><code>class_weight='balanced'</code></p>
            <p>✅ No modifica el dataset original</p>""","green")

    with tab2:
        c1, c2 = st.columns([1, 2.5])
        with c1:
            imb_r = st.slider("% clase minoritaria", 2, 50, 10)
            n_imb = st.slider("Muestras totales", 300, 2000, 600, 100)
            tech  = st.selectbox("Técnica", ["Sin balanceo","SMOTE","Undersampling","class_weight='balanced'"])
            imb_d = st.slider("max_depth", 1, 15, 5, key="imb_d")

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
            vc_i = pd.Series(y_tr_b).value_counts().sort_index()
            ax_i[0].bar(['Clase 0','Clase 1'], vc_i.values, color=['#60a5fa','#fb7185'], edgecolor='none', width=0.5)
            ax_i[0].set_title(f"Distribución train — {tech}")
            for j, v in enumerate(vc_i.values):
                ax_i[0].text(j, v+3, str(int(v)), ha='center', fontweight='bold', color='#e2e8f0')
            cm_i = confusion_matrix(y_te, yp_i)
            sns.heatmap(cm_i, annot=True, fmt='d', cmap='YlOrRd', ax=ax_i[1],
                        xticklabels=['Pred 0','Pred 1'], yticklabels=['Real 0','Real 1'],
                        linewidths=0.5, linecolor='#161b27')
            ax_i[1].set_title("Confusión (Test)")
            plt.tight_layout(); st.pyplot(fig_i, use_container_width=True); plt.close()

        ci = st.columns(4)
        ci[0].metric("Accuracy",         f"{accuracy_score(y_te,yp_i):.3f}")
        ci[1].metric("Precision (cls 1)",f"{precision_score(y_te,yp_i,zero_division=0):.3f}")
        ci[2].metric("Recall (cls 1)",   f"{recall_score(y_te,yp_i,zero_division=0):.3f}", delta="↑ crítico")
        ci[3].metric("F1 (cls 1)",       f"{f1_score(y_te,yp_i,zero_division=0):.3f}")

# ══════════════════════════════════════════════════════════
#  MODULE 7: BAGGING
# ══════════════════════════════════════════════════════════
def show_bagging():
    section_header("🌳","Bagging & Random Forest","Ensamble paralelo para reducir varianza")
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🌲 Random Forest Interactivo", "📊 Feature Importance"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card("""<h4>Bootstrap AGGregating</h4>
            <ol class="step-list" style="margin:0.5rem 0 0">
            <li>Crea B subconjuntos via bootstrapping (con reemplazo)</li>
            <li>Entrena un modelo independiente en cada subconjunto</li>
            <li>Combina: promedio (regresión) o votación (clasificación)</li>
            </ol>
            <p style="margin-top:0.8rem"><b>Objetivo:</b> reducir varianza</p>""", "teal")
        with c2:
            card("""<h4>Random Forest = Bagging + Feature Subsampling</h4>
            <p>En cada split usa solo <b>√n_features</b> features aleatorias →
            descorelaciona los árboles → menor varianza que bagging puro.</p>
            <p><b>OOB Score</b>: ~37% de muestras quedan fuera de cada árbol
            → validación gratuita sin necesitar test set separado.</p>""", "green")

        formula_card("""<b>¿Por qué funciona?</b>  B modelos con varianza σ² y correlación ρ:
Var(ensemble) = ρσ² + (1−ρ)σ²/B

Random Forest reduce ρ al aleatorizar features → mejor que Bagging puro.""")

    with tab2:
        c1, c2 = st.columns([1, 2.5])
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
                go.Scatter(x=n_rng, y=accs_rf, mode='lines+markers', name='Accuracy Test',
                           line=dict(color='#2dd4bf',width=2.5),
                           marker=dict(size=6, color='#2dd4bf', line=dict(color='#0e1117',width=1)))
            ])
            fig_rf.add_vline(x=rf_n, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"n={rf_n}", annotation_font_color="#f59e0b")
            fig_rf.update_layout(xaxis_title="Número de árboles", yaxis_title="Accuracy",
                                  title="Accuracy vs Número de Árboles", height=350,
                                  margin=dict(t=50,b=40,l=40,r=20), **PLOTLY_BASE)
            st.plotly_chart(fig_rf, use_container_width=True)

        cr = st.columns(4 if rf_b else 3)
        cr[0].metric("Accuracy (Test)", f"{accuracy_score(y_te,yp_rf):.4f}")
        cr[1].metric("F1 (macro)",      f"{f1_score(y_te,yp_rf,average='macro'):.4f}")
        cr[2].metric("Árboles",          rf_n)
        if rf_b: cr[3].metric("OOB Score", f"{rf.oob_score_:.4f}")

        single = DecisionTreeClassifier(max_depth=rf_md, random_state=42).fit(X_tr,y_tr)
        imp_val = accuracy_score(y_te,yp_rf) - single.score(X_te,y_te)
        kind = "ok" if imp_val >= 0 else "warn"
        alert(f"🌲 Árbol solo: <b>{single.score(X_te,y_te):.4f}</b> · 🌳🌳 Random Forest: <b>{accuracy_score(y_te,yp_rf):.4f}</b> · Mejora: <b>{imp_val:+.4f}</b>", kind)

    with tab3:
        c1, c2 = st.columns([1, 2])
        with c1:
            fi_ds_rf = st.selectbox("Dataset", ["🎗️ Breast Cancer","🍷 Wine (vinos)","🌸 Iris (flores)"], key="fi_rf_ds")
            fi_n_rf  = st.slider("n_estimators", 10, 300, 100, key="fi_rf_n")
        X_fi, y_fi, fn_fi, _ = load_dataset(fi_ds_rf)
        X_tr_fi, _, y_tr_fi, _ = train_test_split(X_fi, y_fi, test_size=0.3, random_state=42, stratify=y_fi)
        rf_fi = RandomForestClassifier(n_estimators=fi_n_rf, random_state=42, n_jobs=-1).fit(X_tr_fi, y_tr_fi)
        imp_fi  = rf_fi.feature_importances_
        stds_fi = np.std([t.feature_importances_ for t in rf_fi.estimators_], axis=0)
        sidx_fi = np.argsort(imp_fi)[::-1][:15]
        with c2:
            fig_fi_r, ax_fi_r = plt.subplots(figsize=(9,6))
            colors_fi_r = ['#2dd4bf' if i==sidx_fi[0] else '#1a9fa0' if i in sidx_fi[:5] else '#164e63' for i in sidx_fi]
            ax_fi_r.barh(range(len(sidx_fi)), imp_fi[sidx_fi], xerr=stds_fi[sidx_fi],
                          color=colors_fi_r, edgecolor='none', error_kw=dict(ecolor='#475569',capsize=3))
            ax_fi_r.set_yticks(range(len(sidx_fi)))
            ax_fi_r.set_yticklabels([fn_fi[i] for i in sidx_fi], fontsize=9)
            ax_fi_r.set_xlabel("Importancia media ± σ")
            ax_fi_r.set_title(f"Feature Importance — RF ({fi_n_rf} árboles)")
            plt.tight_layout(); st.pyplot(fig_fi_r, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════
#  MODULE 8: BOOSTING
# ══════════════════════════════════════════════════════════
def show_boosting():
    section_header("🚀","Boosting & Gradient Boosting","Ensamble secuencial que reduce sesgo")
    tab1, tab2, tab3 = st.tabs(["📖 Teoría", "🎮 Gradient Boosting Interactivo", "⚡ Comparación Algoritmos"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            card("""<h4>Boosting — Entrenamiento Secuencial</h4>
            <ol class="step-list" style="margin:0.5rem 0 0">
            <li>Entrena modelo₁ con los datos originales</li>
            <li>Identifica los errores del modelo₁</li>
            <li>Entrena modelo₂ dando más peso a los errores</li>
            <li>Repite B veces acumulando correcciones</li>
            <li>Predicción final = combinación ponderada</li>
            </ol>
            <p style="margin-top:0.8rem"><b>Objetivo:</b> reducir sesgo (underfitting)</p>""", "teal")
        with c2:
            card("""<h4>Gradient Boosting</h4>
            <p>Cada árbol predice el <b>residual</b> del ensamble anterior:</p>
            <p>r₁ = y − ŷ₁ &nbsp;→&nbsp; modelo₂ predice r₁<br>
            F(x) = η · Σ fₜ(x) &nbsp;(η = learning rate)</p>
            <p><b>Variantes:</b> AdaBoost, GBM, XGBoost, LightGBM, CatBoost</p>""", "amber")

        formula_card("""<b>Bagging vs Boosting:</b>
Bagging   → paralelo  · modelos independientes · reduce varianza
Boosting  → secuencial · modelos dependientes  · reduce sesgo""")

    with tab2:
        c1, c2 = st.columns([1, 2.5])
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
        staged_tr = [accuracy_score(y_tr, yp) for yp in gb_m.staged_predict(X_tr)]
        staged_te = [accuracy_score(y_te, yp) for yp in gb_m.staged_predict(X_te)]
        best_it   = int(np.argmax(staged_te))

        with c2:
            fig_gb = go.Figure([
                go.Scatter(y=staged_tr, mode='lines', name='Train',
                           line=dict(color='#fb7185',width=2.5)),
                go.Scatter(y=staged_te, mode='lines', name='Test',
                           line=dict(color='#2dd4bf',width=2.5)),
            ])
            fig_gb.add_vline(x=best_it, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Mejor iter={best_it}",
                              annotation_font_color="#f59e0b")
            fig_gb.update_layout(xaxis_title="Iteración (árboles añadidos)", yaxis_title="Accuracy",
                                  title=f"Staged Score — LR={gb_lr}, depth={gb_d}",
                                  height=380, margin=dict(t=50,b=40,l=40,r=20), **PLOTLY_BASE)
            st.plotly_chart(fig_gb, use_container_width=True)

        cg = st.columns(4)
        cg[0].metric("Mejor Acc (Test)", f"{max(staged_te):.4f}")
        cg[1].metric("Mejor iteración",  best_it)
        cg[2].metric("Acc final (Test)", f"{staged_te[-1]:.4f}")
        cg[3].metric("Árboles totales",  gb_n)

        if max(staged_te) > staged_te[-1]+0.01:
            alert(f"<b>Overfitting detectado</b> — la mejor iteración fue {best_it}. "
                  f"Considera reducir <code>learning_rate</code> o usar <code>n_estimators={best_it}</code>.", "warn")

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
        pal_c = ['#2dd4bf','#34d399','#60a5fa','#f59e0b','#fb7185']
        fig_c = go.Figure([
            go.Bar(x=df_c['Modelo'], y=df_c['Acc. Test'],  name='Accuracy Test',
                   marker=dict(color=pal_c, opacity=0.9)),
            go.Bar(x=df_c['Modelo'], y=df_c['F1 (macro)'], name='F1 (macro)',
                   marker=dict(color=pal_c, opacity=0.5)),
        ])
        lo_y = max(0, df_c['F1 (macro)'].min()-0.05)
        fig_c.update_layout(barmode='group', yaxis=dict(range=[lo_y,1.0]),
                             height=380, margin=dict(t=20,b=60,l=40,r=20), **PLOTLY_BASE)
        st.plotly_chart(fig_c, use_container_width=True)
        st.dataframe(df_c.set_index('Modelo').round(4).style.background_gradient(
            subset=['Acc. Test','CV Mean'], cmap='YlOrRd'), use_container_width=True)

# ══════════════════════════════════════════════════════════
#  MODULE 9: COMPARISON
# ══════════════════════════════════════════════════════════
def show_comparison():
    section_header("🏆","Comparación Completa de Modelos","Benchmark interactivo con validación cruzada")
    alert("Selecciona dataset y parámetros. Todos los modelos se evalúan con K-Fold CV para una comparación justa.", "info")

    with st.expander("📦 Ver descripción de todos los datasets"):
        for name, info in DATASET_INFO.items():
            tag = "Binario" if info["binary"] else "Multi-clase"
            st.markdown(f"**{name}** `{tag}` — {info['desc']}")

    c1, c2, c3 = st.columns(3)
    with c1: bench_ds = st.selectbox("Dataset", list(DATASET_INFO.keys()), key="bench_ds")
    with c2: bench_cv = st.slider("K-Fold CV", 3, 10, 5, key="bench_cv")
    with c3: bench_n  = st.slider("Estimadores", 20, 200, 100, key="bench_n")

    X_b, y_b, _, _ = load_dataset(bench_ds)
    X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.3, random_state=42, stratify=y_b)
    bench_models = {
        "DT depth=2":     DecisionTreeClassifier(max_depth=2,    random_state=42),
        "DT depth=5":     DecisionTreeClassifier(max_depth=5,    random_state=42),
        "DT full":        DecisionTreeClassifier(max_depth=None, random_state=42),
        "Bagging":        BaggingClassifier(n_estimators=bench_n, random_state=42, n_jobs=-1),
        "Random Forest":  RandomForestClassifier(n_estimators=bench_n, random_state=42, n_jobs=-1),
        "AdaBoost":       make_adaboost(bench_n),
        "Grad. Boosting": GradientBoostingClassifier(n_estimators=bench_n, random_state=42),
    }
    bench_rows = []
    with st.spinner("Entrenando y evaluando todos los modelos…"):
        for name, m in bench_models.items():
            m.fit(X_tr, y_tr); yp = m.predict(X_te)
            tr_a = m.score(X_tr,y_tr); te_a = accuracy_score(y_te,yp)
            cv = cross_val_score(m, X_b, y_b, cv=bench_cv, scoring='accuracy', n_jobs=-1)
            bench_rows.append({'Modelo':name,'Train Acc':round(tr_a,4),
                                'Test Acc':round(te_a,4),
                                'F1 (macro)':round(f1_score(y_te,yp,average='macro'),4),
                                'CV Mean':round(cv.mean(),4),'CV Std':round(cv.std(),4),
                                'Gap':round(tr_a-te_a,4)})
    df_b = pd.DataFrame(bench_rows).sort_values('CV Mean', ascending=False)

    tab_a, tab_b, tab_c = st.tabs(["📊 Resultados", "🫧 Scatter Plot", "🕸️ Radar Chart"])

    with tab_a:
        st.dataframe(
            df_b.set_index('Modelo').style
                .background_gradient(subset=['CV Mean','Test Acc'], cmap='YlOrRd')
                .background_gradient(subset=['Gap'], cmap='RdYlGn_r'),
            use_container_width=True)
        best = df_b.iloc[0]
        alert(f"🏆 <b>Mejor modelo (CV Mean):</b> {best['Modelo']} — {best['CV Mean']:.4f} ± {best['CV Std']:.4f}", "ok")

    with tab_b:
        fig_sc = px.scatter(df_b, x='CV Std', y='CV Mean', text='Modelo', size='Test Acc',
                             color='Gap', color_continuous_scale='RdYlGn_r',
                             title='CV Accuracy vs Inestabilidad (tamaño = Test Acc)',
                             labels={'CV Std':'Inestabilidad (σ)','CV Mean':'Accuracy CV Promedio'})
        fig_sc.update_traces(textposition='top center', textfont_size=10, textfont_color='#e2e8f0')
        fig_sc.update_layout(height=440, margin=dict(t=50,b=40), **PLOTLY_BASE)
        st.plotly_chart(fig_sc, use_container_width=True)

    with tab_c:
        cats = ['Test Acc','F1 (macro)','CV Mean']
        pal_r = ['#2dd4bf','#fb7185','#34d399','#f59e0b','#60a5fa','#a78bfa','#fb923c']
        fig_r = go.Figure()
        for i, (_, row) in enumerate(df_b.iterrows()):
            vals = [row['Test Acc'], row['F1 (macro)'], row['CV Mean']]
            fig_r.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                name=row['Modelo'], fill='toself', opacity=0.45,
                line=dict(color=pal_r[i%len(pal_r)], width=2)))
        lo_r = max(0.5, df_b[cats].values.min()-0.05)
        fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[lo_r,1.0],
                                        gridcolor='#2d3748', linecolor='#2d3748')),
                             title="Radar de Desempeño Multidimensional",
                             height=480, paper_bgcolor='rgba(0,0,0,0)',
                             font=dict(family='DM Sans, sans-serif', color='#e2e8f0'))
        st.plotly_chart(fig_r, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🗃️ Benchmark Multi-Dataset")
    multi_ds = st.multiselect("Selecciona datasets:", list(DATASET_INFO.keys()),
                               default=["🌸 Iris (flores)","🎗️ Breast Cancer","🍷 Wine (vinos)"])
    if multi_ds and st.button("🚀 Lanzar benchmark multi-dataset", type="primary"):
        quick_models = {
            "DT depth=5":    DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Grad. Boosting":GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        qrows = []
        with st.spinner("Calculando…"):
            for ds_name in multi_ds:
                X_q, y_q, _, _ = load_dataset(ds_name)
                for mname, m in quick_models.items():
                    sc = cross_val_score(m, X_q, y_q, cv=5, scoring='accuracy', n_jobs=-1)
                    qrows.append({'Dataset':ds_name,'Modelo':mname,
                                   'CV Mean':round(sc.mean(),4),'CV Std':round(sc.std(),4)})
        df_q = pd.DataFrame(qrows)
        fig_q = px.bar(df_q, x='Dataset', y='CV Mean', color='Modelo', barmode='group',
                        error_y='CV Std', title="Accuracy CV por Dataset y Modelo",
                        color_discrete_sequence=['#2dd4bf','#34d399','#f59e0b'])
        fig_q.update_layout(height=420, margin=dict(t=50,b=80), **PLOTLY_BASE)
        st.plotly_chart(fig_q, use_container_width=True)
        pivot_q = df_q.pivot(index='Dataset', columns='Modelo', values='CV Mean')
        st.dataframe(pivot_q.style.background_gradient(cmap='YlOrRd'), use_container_width=True)

# ══════════════════════════════════════════════════════════
#  MODULE 10: LAB
# ══════════════════════════════════════════════════════════
def show_lab():
    section_header("🔬","Laboratorio Libre","Experimenta con cualquier modelo y dataset")
    alert("Elige modelo, dataset e hiperparámetros. Pulsa <b>Entrenar</b> para ver los resultados.", "info")

    c1, c2, c3 = st.columns(3)
    with c1: lab_model = st.selectbox("Modelo", ["Decision Tree","Random Forest","AdaBoost","Gradient Boosting","Bagging"])
    with c2: lab_ds    = st.selectbox("Dataset", list(DATASET_INFO.keys()))
    with c3: lab_ts    = st.slider("Test size", 0.1, 0.5, 0.3, 0.05)

    X_lab, y_lab, fn_lab, cn_lab = load_dataset(lab_ds)
    st.markdown("**⚙️ Hiperparámetros**")
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
        with pc[2]: p3 = st.slider("max_features_ratio",0.3,1.0,1.0,0.1,key="lp3")
        model_lab = BaggingClassifier(n_estimators=p1,max_samples=p2,max_features=p3,random_state=42,n_jobs=-1)

    if st.button("🚀 Entrenar y Evaluar", type="primary"):
        with st.spinner("Entrenando…"):
            X_tr, X_te, y_tr, y_te = train_test_split(X_lab, y_lab, test_size=lab_ts, random_state=42, stratify=y_lab)
            model_lab.fit(X_tr, y_tr); yp = model_lab.predict(X_te)
            cv_sc = cross_val_score(model_lab, X_lab, y_lab, cv=5, scoring='accuracy', n_jobs=-1)

        tr_a = model_lab.score(X_tr,y_tr); te_a = accuracy_score(y_te,yp); gap_l = tr_a - te_a
        cr = st.columns(5)
        cr[0].metric("Train Acc",  f"{tr_a:.4f}")
        cr[1].metric("Test Acc",   f"{te_a:.4f}")
        cr[2].metric("F1 (macro)", f"{f1_score(y_te,yp,average='macro'):.4f}")
        cr[3].metric("CV Mean",    f"{cv_sc.mean():.4f}")
        cr[4].metric("CV Std",     f"{cv_sc.std():.4f}")

        c1, c2 = st.columns([1,1])
        with c1:
            fig_l, ax_l = plt.subplots(figsize=(5,4))
            sns.heatmap(confusion_matrix(y_te,yp), annot=True, fmt='d', cmap='YlOrRd',
                        ax=ax_l, xticklabels=[str(c) for c in cn_lab],
                        yticklabels=[str(c) for c in cn_lab],
                        linewidths=0.5, linecolor='#161b27')
            ax_l.set_title(f"Matriz de Confusión"); ax_l.set_xlabel("Predicho"); ax_l.set_ylabel("Real")
            st.pyplot(fig_l); plt.close()
        with c2:
            st.markdown("**Classification Report**")
            st.code(classification_report(y_te, yp, target_names=[str(c) for c in cn_lab]), language="text")

        diagnosis_badge(gap_l, te_a)

# ══════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════
routes = {
    "home":           show_home,
    "bias_variance":  show_bias_variance,
    "decision_trees": show_decision_trees,
    "entropy":        show_entropy,
    "metrics":        show_metrics,
    "validation":     show_validation,
    "imbalanced":     show_imbalanced,
    "bagging":        show_bagging,
    "boosting":       show_boosting,
    "comparison":     show_comparison,
    "lab":            show_lab,
}
routes.get(module, show_home)()
