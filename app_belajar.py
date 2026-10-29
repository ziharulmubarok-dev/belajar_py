import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Analisis Big Data & Machine Learning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style Custom CSS Singkat
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Aplikasi Analisis Big Data & Klasifikasi")
st.write("Unggah dataset Anda untuk melatih model Machine Learning dan melakukan evaluasi performa secara dinamis.")

# ==========================================
# SIDEBAR: UPLOAD DATA & PENGATURAN MODEL
# ==========================================
st.sidebar.header("📁 1. Unggah Data")

uploaded_train = st.sidebar.file_uploader("Unggah Data Training (CSV / Excel)", type=["csv", "xlsx"])
uploaded_test = st.sidebar.file_uploader("Unggah Data Testing (Opsional)", type=["csv", "xlsx"])

df_train = None
df_test = None

# Load Training Data
if uploaded_train is not None:
    if uploaded_train.name.endswith('.csv'):
        df_train = pd.read_csv(uploaded_train)
    else:
        df_train = pd.read_excel(uploaded_train)

# Load Testing Data (Jika ada)
if uploaded_test is not None:
    if uploaded_test.name.endswith('.csv'):
        df_test = pd.read_csv(uploaded_test)
    else:
        df_test = pd.read_excel(uploaded_test)

# ==========================================
# MAIN PAGE CONTENT
# ==========================================
if df_train is not None:
    st.subheader("📋 Ringkasan Dataset")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Preview Data Training:**")
        st.dataframe(df_train.head(), use_container_width=True)
        st.caption(f"Dimensi Data Training: {df_train.shape[0]} baris, {df_train.shape[1]} kolom")
        
    with col2:
        if df_test is not None:
            st.write("**Preview Data Testing:**")
            st.dataframe(df_test.head(), use_container_width=True)
            st.caption(f"Dimensi Data Testing: {df_test.shape[0]} baris, {df_test.shape[1]} kolom")
        else:
            st.info("💡 Data Testing terpisah tidak diunggah. Aplikasi akan membagi Data Training secara otomatis (Train-Test Split).")

    st.markdown("---")
    st.sidebar.header("⚙️ 2. Parameter Model")
    
    # Pilih Target Variable
    columns = df_train.columns.tolist()
    target_col = st.sidebar.selectbox("Pilih Kolom Target (Label/Class):", columns, index=len(columns)-1)
    
    # Pilih Feature Columns
    feature_cols = st.sidebar.multiselect(
        "Pilih Kolom Fitur (Prediktor):", 
        [col for col in columns if col != target_col],
        default=[col for col in columns if col != target_col]
    )

    if not feature_cols:
        st.warning("⚠️ Silakan pilih setidaknya satu kolom fitur untuk melanjutkan.")
        st.stop()

    # Parameter Model Hyperparameters
    n_estimators = st.sidebar.slider("Jumlah Trees (n_estimators):", 10, 200, 100, step=10)
    max_depth = st.sidebar.slider("Kedalaman Maksimum (max_depth):", 2, 20, 10)
    
    if df_test is None:
        test_size = st.sidebar.slider("Rasio Split Data Testing:", 0.1, 0.4, 0.2, step=0.05)

    # Tombol Eksekusi
    if st.sidebar.button("🚀 Jalankan Pemodelan"):
        st.subheader("⚡ Pemrosesan Data & Pelatihan Model")
        
        # Preprocessing Data
        X_train_raw = df_train[feature_cols].copy()
        y_train_raw = df_train[target_col].copy()

        # Handling Categorical Encoding
        le = LabelEncoder()
        for col in X_train_raw.select_dtypes(include=['object', 'category']).columns:
            X_train_raw[col] = le.fit_transform(X_train_raw[col].astype(str))
            
        if y_train_raw.dtype == 'object' or y_train_raw.dtype == 'category':
            y_train_raw = le.fit_transform(y_train_raw.astype(str))

        if df_test is not None:
            X_test_raw = df_test[feature_cols].copy()
            y_test_raw = df_test[target_col].copy()
            
            for col in X_test_raw.select_dtypes(include=['object', 'category']).columns:
                X_test_raw[col] = le.fit_transform(X_test_raw[col].astype(str))
                
            if y_test_raw.dtype == 'object' or y_test_raw.dtype == 'category':
                y_test_raw = le.fit_transform(y_test_raw.astype(str))

            X_tr, X_te = X_train_raw, X_test_raw
            y_tr, y_te = y_train_raw, y_test_raw
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_raw, y_train_raw, test_size=test_size, random_state=42
            )

        # Feature Scaling
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Fitting Random Forest Model
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_tr_scaled, y_tr)
        
        # Prediksi
        y_pred = model.predict(X_te_scaled)

        # Metrik Evaluasi
        acc = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_te, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_te, y_pred, average='weighted', zero_division=0)

        # Tampilan Hasil Evaluasi
        st.subheader("📈 Evaluasi Performa Model")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Akurasi", f"{acc * 100:.2f}%")
        m2.metric("Precision", f"{prec * 100:.2f}%")
        m3.metric("Recall", f"{rec * 100:.2f}%")
        m4.metric("F1-Score", f"{f1 * 100:.2f}%")

        st.markdown("---")

        # Visualisasi Confusion Matrix & Feature Importance
        col_vis1, col_vis2 = st.columns(2)

        with col_vis1:
            st.write("**Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_te, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Prediksi")
            ax_cm.set_ylabel("Aktual")
            st.pyplot(fig_cm)

        with col_vis2:
            st.write("**Feature Importance**")
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({'Fitur': feature_cols, 'Tingkat Kepentingan': importances}).sort_values('Tingkat Kepentingan', ascending=True)
            
            fig_imp, ax_imp = plt.subplots(figsize=(5, 4))
            ax_imp.barh(feature_imp_df['Fitur'], feature_imp_df['Tingkat Kepentingan'], color='#1f77b4')
            ax_imp.set_xlabel("Nilai Importansi")
            st.pyplot(fig_imp)

else:
    st.info("👋 Selamat Datang! Silakan unggah file data training melalui *sidebar* di sebelah kiri untuk memulai analisis.")