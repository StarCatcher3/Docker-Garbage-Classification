import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import requests

# --- CONFIGURATION DES CHEMINS ---

API_URL = "http://api-service:8000"

CLASSES = ["biodegradable", "cardboard", "glass", "metal", "paper", "plastic"]
CANVA_GREEN = "#1E7046"
CANVA_BEIGE = "#D6D1C1"

st.set_page_config(page_title="Garbage Analysis Dashboard", layout="wide")

# --- STYLE CSS (VERT SOMBRE & BEIGE) ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {CANVA_BEIGE}; }}
    header[data-testid="stHeader"] {{ background-color: {CANVA_GREEN} !important; }}

    /* Zone d'Upload - Rectangle Vert */
    [data-testid="stFileUploader"] section {{
        background-color: {CANVA_GREEN} !important;
        border: 2px dashed #FFFFFF !important;
        border-radius: 10px !important;
    }}
    /* Texte blanc dans l'uploader pour la visibilité */
    [data-testid="stFileUploader"] section div, [data-testid="stFileUploader"] section p, [data-testid="stFileUploader"] section span {{
        color: #FFFFFF !important;
    }}
    [data-testid="stFileUploader"] section .stButton > button {{
        background-color: #FFFFFF !important;
        color: {CANVA_GREEN} !important;
    }}

    /* Toute l'écriture en Vert Sombre */
    .stMarkdown, p, label, .stText, h1, h2, h3, span {{
        color: {CANVA_GREEN} !important;
    }}

    /* Métriques et Tableaux */
    div[data-testid="stMetricValue"] {{
        background-color: #FFFFFF;
        color: {CANVA_GREEN} !important;
        border: 1px solid {CANVA_GREEN};
        border-radius: 10px;
        padding: 10px;
    }}
    .stTable {{ background-color: #FFFFFF; border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)

response = requests.get(f"{API_URL}/data")
result = response.json()

df_preds = pd.DataFrame(result["data"]) if response.status_code == 200 else pd.DataFrame()


st.title("♻️ Garbage Classification Dashboard")
st.header("📌 Statistiques du Dataset")
c_met, c_pie = st.columns([3, 2])

with c_met:
    m1, m2 = st.columns(2)
    m1.metric("Total Images (Dataset)", 10464)  # Valeur fixe demandée
    m1.metric("Nb Classes", 6)
    m2.metric("Images Train", 7324)
    m2.metric("Images Test", 3140)

with c_pie:
    df_split = pd.DataFrame({"Usage": ["Train", "Test"], "Valeur": [7324, 3140]})
    fig_pie = px.pie(df_split, values='Valeur', names='Usage', hole=0.4,
                     color_discrete_sequence=[CANVA_GREEN, "#7DBE6F"])
    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=CANVA_GREEN, margin=dict(t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)


st.divider()
st.header("🎯 Analyse de la Confiance (Modèle)")

if not df_preds.empty and 'confidence' in df_preds.columns:
    col_t, col_m = st.columns([1, 2])  # 1/3 pour le tableau, 2/3 pour les metrics

    with col_t:
        st.write("**Statistiques complètes :**")
        stats = df_preds['confidence'].describe().to_frame()
        stats.columns = ['Valeurs']
        st.table(stats.style.format("{:.4f}"))

    with col_m:
        st.write("**Indicateurs clés :**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Moyenne", f"{df_preds['confidence'].mean():.2%}")
        m2.metric("Médiane (50%)", f"{df_preds['confidence'].median():.2%}")
        m3.metric("Écart-type (Std)", f"{df_preds['confidence'].std():.4f}")

        st.write("---")

        m4, m5 = st.columns(2)
        m4.metric("Minimum", f"{df_preds['confidence'].min():.2%}")
        m5.metric("Maximum", f"{df_preds['confidence'].max():.2%}")

        st.info("💡 L'écart-type (std) mesure la stabilité.")
else:
    st.info("Aucune donnée de confiance disponible. Lancez le script de prédiction Spark.")

if not df_preds.empty and 'confidence' in df_preds.columns and ('class' in df_preds.columns or 'category' in df_preds.columns):
    st.write("---")
    st.write("**Confiance moyenne par catégorie :**")

    label_col = "class" if "class" in df_preds.columns else "category"
    df_avg_conf = df_preds.groupby(label_col)['confidence'].mean().reset_index()
    df_avg_conf = df_avg_conf.sort_values('confidence', ascending=False)

    fig_avg = px.bar(
        df_avg_conf,
        x=label_col,
        y="confidence",
        color=label_col,
        text_auto='.2%',
        title="Précision moyenne du modèle par classe",
        labels={label_col: "Catégorie", "confidence": "Confiance Moyenne"}
    )

    fig_avg.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=CANVA_GREEN,
        xaxis=dict(showgrid=False, color=CANVA_GREEN),
        yaxis=dict(showgrid=True, gridcolor="#CCCCCC", color=CANVA_GREEN, tickformat='.0%'),
        margin=dict(t=50, b=50),
        showlegend=False
    )

    fig_avg.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_avg, use_container_width=True)

st.divider()
st.header("📤 Inférence & Gestion des Flux")
uploaded_files = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    cols = st.columns(3)
    for i, file in enumerate(uploaded_files):
        with cols[i % 3]:
            image_archive = requests.post(
                f"{API_URL}/get",
                files={"file": (file.name, file.getvalue())}
            )

            if image_archive.status_code == 404:          
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (file.name, file.getvalue())}
                )

                if response.status_code == 200:
                    img = Image.open(file)
                    st.image(img, use_container_width=True)
                    st.success(f"{file.name} uploaded")
                else:
                    st.error("Upload failed")
            else:
                data = image_archive.json()
                
                img = Image.open(file)
                st.image(img, use_container_width=True)
                st.success(f"### Résultat : {data['class_name']} ({data['confidence']:.2%}) (Déjà archivé)")