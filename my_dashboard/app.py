# superstore_dashboard/app_enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import chardet
import os
import kagglehub
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler


# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PALETTE DE COULEURS PERSONNALISÉE
# ============================================
COLORS = {
    'profit': '#2ca02c',      # Vert pour le profit
    'loss': '#d62728',         # Rouge pour les pertes
    'sales': '#1f77b4',        # Bleu pour les ventes
    'background': "#447be9",
    'card_bg': '#ffffff',
    'text_dark': '#262730',
    'text_light': '#ffffff',
    'accent': '#ff4b4b'
}

# Style CSS personnalisé
st.markdown(f"""
    <style>
    /* Style général */
    .stApp {{
        background-color: {COLORS['background']};
    }}
    
    /* En-tête principal */
    .main-header {{
        font-size: 3rem;
        color: {COLORS['accent']};
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Cartes KPI */
    .kpi-card {{
        background: linear-gradient(135deg, {COLORS['sales']} 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: {COLORS['text_light']};
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s;
    }}
    .kpi-card:hover {{
        transform: translateY(-5px);
    }}
    .kpi-card-profit {{
        background: linear-gradient(135deg, {COLORS['profit']} 0%, #1e7e1e 100%);
    }}
    .kpi-card-loss {{
        background: linear-gradient(135deg, {COLORS['loss']} 0%, #a12323 100%);
    }}
    .kpi-card-sales {{
        background: linear-gradient(135deg, {COLORS['sales']} 0%, #0e4b75 100%);
    }}
    .kpi-title {{
        font-size: 1.1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }}
    .kpi-delta {{
        font-size: 0.9rem;
        margin-top: 0.3rem;
        opacity: 0.8;
    }}
    
    /* Sections */
    .section-header {{
        font-size: 2rem;
        color: {COLORS['text_dark']};
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 4px solid {COLORS['accent']};
        font-weight: 600;
    }}
    .subsection-header {{
        font-size: 1.5rem;
        color: {COLORS['text_dark']};
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid {COLORS['sales']};
    }}
    
    /* Cartes d'information */
    .info-card {{
        background-color: {COLORS['card_bg']};
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 8px solid {COLORS['accent']};
    }}
    .info-card h3 {{
        color: {COLORS['text_dark']};
        margin-top: 0;
        font-size: 1.8rem;
    }}
    .info-card h4 {{
        color: {COLORS['text_dark']};
        margin-top: 0;
        font-size: 1.4rem;
    }}
    .info-card p {{
        color: {COLORS['text_dark']};
        font-size: 1.2rem;
        line-height: 1.8;
    }}
    
    /* Métriques spéciales */
    .profit-text {{
        color: {COLORS['profit']};
        font-weight: bold;
    }}
    .loss-text {{
        color: {COLORS['loss']};
        font-weight: bold;
    }}
    .sales-text {{
        color: {COLORS['sales']};
        font-weight: bold;
    }}
    
    /* Cartes d'objectifs */
    .objective-card {{
        background-color: {COLORS['card_bg']};
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 100%;
        border-top: 5px solid {COLORS['sales']};
    }}
    .objective-card h4 {{
        color: {COLORS['sales']};
        font-size: 1.5rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid {COLORS['sales']};
        padding-bottom: 0.5rem;
    }}
    .objective-card ul {{
        color: {COLORS['text_dark']};
        font-size: 1.1rem;
        line-height: 2.2;
        padding-right: 1rem;
    }}
    
    .method-card {{
        background-color: {COLORS['card_bg']};
        padding: 1.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        height: 100%;
        border-top: 5px solid {COLORS['profit']};
    }}
    .method-card h4 {{
        color: {COLORS['profit']};
        font-size: 1.5rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid {COLORS['profit']};
        padding-bottom: 0.5rem;
    }}
    .method-card ul {{
        color: {COLORS['text_dark']};
        font-size: 1.1rem;
        line-height: 2.2;
        padding-right: 1rem;
    }}
    
    /* Cartes de statistiques */
    .stat-card {{
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 5px solid;
        color: {COLORS['text_dark']};
    }}
    .stat-card strong {{
        font-size: 1.2rem;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================
# DICTIONNAIRE ÉTATS → CODES USPS
# ============================================
US_STATE_ABBREV = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL",
    "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

# ============================================
# FONCTIONS DE CHARGEMENT
# ============================================
@st.cache_data
def load_and_clean_data():
    """Charge et nettoie les données du Superstore"""
    
    with st.spinner("📥 Chargement des données..."):
        path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
        csv_file = os.path.join(path, "Sample - Superstore.csv")
        
        with open(csv_file, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw)
        
        df = pd.read_csv(csv_file, encoding=enc['encoding'])
        
        # Nettoyage
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        
        if "row_id" in df.columns:
            df = df.drop(columns=["row_id"])
        
        df['postal_code'] = df['postal_code'].astype(str)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
        df['sales'] = df['sales'].astype(float)
        df['discount'] = df['discount'].astype(float)
        
        df = df.drop_duplicates()
        
        # Feature engineering
        df['shipping_time_days'] = (df["ship_date"] - df['order_date']).dt.days
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['year_quarter'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
        
        mois_fr = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }
        df['month_name'] = df['month'].map(mois_fr)
        
        df['discount_bin'] = pd.cut(
            df['discount'],
            bins=[-0.01, 0, 0.1, 0.2, 0.3, 1],
            labels=['0%', '0–10%', '10–20%', '20–30%', '30%+']
        )
        
        df['is_loss'] = df['profit'] < 0
        
        return df

@st.cache_data
def prepare_clustering_data(df):
    """Prépare les données pour le clustering"""
    client_features = (
        df.groupby("customer_id")
        .agg({
            "profit": "sum",
            "order_id": "nunique",
            "sales": "mean",
            "discount": "mean"
        })
        .rename(columns={
            "profit": "total_profit",
            "order_id": "nb_orders",
            "sales": "avg_basket",
            "discount": "avg_discount"
        })
        .round(2)
    )
    return client_features

@st.cache_data
def perform_clustering(client_features, n_clusters=4):
    """Réalise le clustering K-Means"""
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(client_features)
    
    iso = IsolationForest(contamination=0.03, random_state=42)
    outliers = iso.fit_predict(X_scaled)
    client_features_with_outliers = client_features.copy()
    client_features_with_outliers["outlier"] = outliers
    
    X_clean = X_scaled[outliers == 1]
    client_clean = client_features[outliers == 1].copy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clean)
    client_clean["cluster"] = clusters
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_clean)
    client_clean["pca_1"] = X_pca[:, 0]
    client_clean["pca_2"] = X_pca[:, 1]
    client_clean["pca_3"] = X_pca[:, 2]
    
    return client_clean, client_features_with_outliers, pca.explained_variance_ratio_, X_clean, kmeans

# ============================================
# FONCTIONS DE VISUALISATION AVANCÉES
# ============================================

def plot_cluster_heatmap(client_clean):
    """Heatmap des profils de clusters avec min/max highlighting"""
    cols = ["total_profit", "nb_orders", "avg_basket", "avg_discount"]
    cluster_profile = (
        client_clean
        .groupby("cluster")[cols]
        .mean()
        .round(2)
    )
    
    max_mask = cluster_profile.eq(cluster_profile.max())
    min_mask = cluster_profile.eq(cluster_profile.min())
    
    colorscale = [
        [0, "#f7fbff"],
        [0.5, "#c6dbef"],
        [1, "#6baed6"]
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=cluster_profile.values,
        x=cluster_profile.columns,
        y=cluster_profile.index,
        colorscale=colorscale,
        text=cluster_profile.values,
        texttemplate="%{text}",
        colorbar=dict(title="Valeur"),
        hovertemplate="Cluster %{y}<br>%{x}: %{text}<extra></extra>"
    ))
    
    for i in range(cluster_profile.shape[0]):
        for j in range(cluster_profile.shape[1]):
            if max_mask.iloc[i, j]:
                fig.add_shape(
                    type="rect",
                    x0=j-0.5, x1=j+0.5,
                    y0=i-0.5, y1=i+0.5,
                    line=dict(color=COLORS['profit'], width=3),
                    fillcolor="rgba(0,0,0,0)"
                )
            if min_mask.iloc[i, j]:
                fig.add_shape(
                    type="rect",
                    x0=j-0.5, x1=j+0.5,
                    y0=i-0.5, y1=i+0.5,
                    line=dict(color=COLORS['loss'], width=3),
                    fillcolor="rgba(0,0,0,0)"
                )
    
    fig.update_layout(
        title="Profil réel des clusters (K=4)",
        title_x=0.5,
        height=400,
        xaxis_title="Variables",
        yaxis_title="Cluster"
    )
    
    return fig

def plot_distribution_cluster(df_cluster, cluster_id):
    """Distribution des variables pour un cluster"""
    variables = ["total_profit", "nb_orders", "avg_basket", "avg_discount"]
    colors = [COLORS['sales'], "#F58518", COLORS['profit'], "#B279A2"]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[v.replace("_", " ").title() for v in variables]
    )
    
    for i, (var, color) in enumerate(zip(variables, colors)):
        row = i // 2 + 1
        col = i % 2 + 1
        data = df_cluster[var].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=data,
                marker_color=color,
                opacity=0.6,
                showlegend=False,
                nbinsx=20,
                name=var
            ),
            row=row, col=col
        )
        
        if len(data) > 1:
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            y_kde = kde(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde * len(data) * (data.max() - data.min()) / 20,
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=600,
        title_text=f"Distribution des variables - Cluster {cluster_id}",
        template="plotly_white",
        bargap=0.15,
        showlegend=False
    )
    
    return fig

def plot_mirror_chart(df_cluster, cluster_id):
    """Graphique miroir ventes vs profits par sous-catégorie"""
    subcat_cluster = (
        df_cluster
        .groupby("subcategory")[["sales", "profit"]]
        .sum()
        .sort_values("sales")
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=subcat_cluster.index,
        x=-subcat_cluster["sales"],
        name="Ventes",
        orientation="h",
        marker_color=COLORS['sales'],
        hovertemplate="Ventes: $%{x:.0f}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        y=subcat_cluster.index,
        x=subcat_cluster["profit"],
        name="Profit",
        orientation="h",
        marker_color=COLORS['profit'],
        hovertemplate="Profit: $%{x:.0f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Cluster {cluster_id} – Ventes vs Profits par Sous-Catégorie",
        barmode="overlay",
        xaxis_title="Montant ($)",
        yaxis_title="Sous-catégorie",
        template="plotly_white",
        height=500,
        hovermode="y unified"
    )
    
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=-0.5, y1=len(subcat_cluster)-0.5,
        line=dict(color=COLORS['loss'], width=2, dash="dash")
    )
    
    return fig

def plot_state_map(df_cluster, cluster_id):
    """Carte choroplèthe des profits par état"""
    state_perf = (
        df_cluster
        .groupby("state")
        .agg({
            'profit': 'sum',
            'sales': 'sum',
            'customer_id': 'nunique',
            'order_id': 'nunique'
        })
        .reset_index()
    )
    
    state_perf['state_code'] = state_perf['state'].map(US_STATE_ABBREV)
    state_perf = state_perf.dropna(subset=['state_code'])
    
    if state_perf.empty:
        return None
    
    fig = px.choropleth(
        state_perf,
        locations='state_code',
        locationmode='USA-states',
        color='profit',
        scope='usa',
        title=f"Cluster {cluster_id} - Profit par État",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        range_color=[-15000, 35000],
        hover_data={
            'state': True,
            'profit': ':,.0f',
            'sales': ':,.0f',
            'customer_id': True,
            'order_id': True
        }
    )
    
    fig.update_layout(
        height=500,
        coloraxis_colorbar_title="Profit ($)",
        template="plotly_white"
    )
    
    return fig

def plot_radar_profile(cluster_profile, cluster_id):
    """Radar chart pour un cluster spécifique"""
    features = ["total_profit", "nb_orders", "avg_basket", "avg_discount"]
    
    scaler = MinMaxScaler()
    cluster_scaled = scaler.fit_transform(cluster_profile[features])
    cluster_scaled_df = pd.DataFrame(
        cluster_scaled,
        columns=features,
        index=cluster_profile.index
    )
    
    values = cluster_scaled_df.loc[cluster_id].values.tolist()
    values += values[:1]
    labels = features + [features[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=f"Cluster {cluster_id}",
        line_color=COLORS['sales'],
        fillcolor=f"rgba(31, 119, 180, 0.3)"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Profil Radar Normalisé - Cluster {cluster_id}",
        template="plotly_white",
        height=400
    )
    
    return fig

def plot_subcategory_analysis(df_clusters):
    """Analyse complète des sous-catégories par cluster"""
    
    # Volume par sous-catégorie
    subcat_count = (
        df_clusters
        .groupby(["cluster", "subcategory"])
        .size()
        .reset_index(name="count")
        .sort_values(["cluster", "count"], ascending=[True, False])
    )
    
    fig_volume = px.bar(
        subcat_count,
        x="subcategory",
        y="count",
        color="cluster",
        facet_row="cluster",
        title="📦 Volume des commandes par sous-catégorie",
        template="plotly_white",
        height=800,
        color_discrete_sequence=[COLORS['sales'], COLORS['profit'], "#ff9800", COLORS['loss']]
    )
    
    fig_volume.update_traces(
        text=subcat_count["count"],
        textposition="outside",
        marker_line_width=1.2
    )
    fig_volume.update_layout(title_x=0.5, bargap=0.25)
    fig_volume.update_xaxes(tickangle=45)
    
    # Profit par sous-catégorie
    subcat_profit = (
        df_clusters
        .groupby(["cluster", "subcategory"])["profit"]
        .sum()
        .reset_index()
        .sort_values(["cluster", "profit"], ascending=[True, False])
    )
    
    fig_profit = px.bar(
        subcat_profit,
        x="subcategory",
        y="profit",
        color="profit",
        facet_row="cluster",
        color_continuous_scale=["#d62728", "#ffffbf", "#2ca02c"],
        title="💰 Profit par sous-catégorie",
        template="plotly_white",
        height=800
    )
    
    fig_profit.update_traces(
        text=subcat_profit["profit"].round(0),
        textposition="outside"
    )
    fig_profit.update_layout(title_x=0.5, bargap=0.25)
    fig_profit.update_xaxes(tickangle=45)
    
    # Taux de perte
    loss_subcat = (
        df_clusters
        .assign(loss=df_clusters["profit"] < 0)
        .groupby(["cluster", "subcategory"])["loss"]
        .mean()
        .reset_index()
        .sort_values(["cluster", "loss"], ascending=[True, False])
    )
    
    fig_loss = px.bar(
        loss_subcat,
        x="subcategory",
        y="loss",
        color="loss",
        facet_row="cluster",
        color_continuous_scale="Reds",
        title="⚠️ Taux de perte par sous-catégorie",
        template="plotly_white",
        height=800
    )
    
    fig_loss.update_traces(
        text=(loss_subcat["loss"] * 100).round(1).astype(str) + "%",
        textposition="outside"
    )
    fig_loss.update_layout(title_x=0.5, bargap=0.25)
    fig_loss.update_yaxes(tickformat=".0%")
    fig_loss.update_xaxes(tickangle=45)
    
    return fig_volume, fig_profit, fig_loss

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
df = load_and_clean_data()
client_features = prepare_clustering_data(df)
client_clean, client_with_outliers, pca_ratio, X_clean, kmeans = perform_clustering(client_features, n_clusters=4)

# Fusion pour analyses détaillées
df_clusters = df.merge(
    client_clean[["cluster"]],
    left_on="customer_id",
    right_index=True,
    how="inner"
)

# Profil des clusters pour les analyses
cols_profile = ["total_profit", "nb_orders", "avg_basket", "avg_discount"]
cluster_profile = (
    client_clean
    .groupby("cluster")[cols_profile]
    .mean()
    .round(2)
)
cluster_profile["nb_clients"] = client_clean.groupby("cluster").size()
# cluster_profile["pct_clients"] = (cluster_profile["nb_clients"] / len(client_clean) * 100).round(1)

# ============================================
# SIDEBAR - FILTRES
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shop.png", width=80)
    st.title("Superstore Analytics Dashboard")
    st.text(" Réalisé par:")
    st.text(" Mohammed SHAQURA")
    st.text(" Rokhaya NDIAYE")
    st.text(" Sarah VELLAYDON")
    st.markdown("---")
    
    st.header("🔍 Filtres principaux")
    
    available_years = sorted(df['year'].unique())
    selected_years = st.multiselect(
        "📅 Années",
        options=available_years,
        default=available_years
    )
    
    available_regions = sorted(df['region'].unique())
    selected_regions = st.multiselect(
        "🌍 Régions",
        options=available_regions,
        default=available_regions
    )
    
    available_categories = sorted(df['category'].unique())
    selected_categories = st.multiselect(
        "🏷️ Catégories",
        options=available_categories,
        default=available_categories
    )
    
    available_segments = sorted(df['segment'].unique())
    selected_segments = st.multiselect(
        "👥 Segment client",
        options=available_segments,
        default=available_segments
    )
    
    st.markdown("---")
    st.header("📊 Aperçu")
    st.metric("Total commandes", f"{len(df):,}")
    st.metric("Clients uniques", f"{df['customer_id'].nunique():,}")
    st.metric("Produits", f"{df['product_name'].nunique():,}")

# Application des filtres
df_filtered = df[
    (df['year'].isin(selected_years)) &
    (df['region'].isin(selected_regions)) &
    (df['category'].isin(selected_categories)) &
    (df['segment'].isin(selected_segments))
]

# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<h1 class="main-header" style="color:black;">📊 Superstore Analytics Dashboard</h1>', unsafe_allow_html=True)

# ============================================
# PAGE D'ACCUEIL (INTRODUCTION)
# ============================================
with st.container():
    # Carte de présentation du projet
    st.markdown("""
    <div class="info-card">
        <h3>🎯 À propos du projet</h3>
        <p>
        Ce tableau de bord interactif présente une analyse complète des ventes du Superstore américain. 
        Il a été développé pour explorer les performances commerciales, comprendre le comportement des clients 
        et identifier les opportunités d'optimisation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        # Objectifs de l'analyse
        st.markdown("""
        <div class="objective-card">
            <h4>📈 Objectifs de l'analyse</h4>
            <ul>
                <li>📊 <strong>Analyser l'évolution</strong> des ventes et des profits dans le temps</li>
                <li>🏷️ <strong>Identifier</strong> les catégories et produits les plus performants</li>
                <li>🌍 <strong>Évaluer la performance</strong> par région géographique</li>
                <li>👥 <strong>Segmenter les clients</strong> en groupes homogènes (K-Means)</li>
                <li>🚚 <strong>Étudier l'impact logistique</strong> sur la rentabilité</li>
                <li>💰 <strong>Comprendre l'effet</strong> des remises sur les profits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Méthodologie utilisée
        st.markdown("""
        <div class="method-card">
            <h4>🔬 Méthodologie</h4>
            <ul>
                <li>🧹 <strong>Nettoyage et préparation</strong> des données</li>
                <li>📊 <strong>Analyse exploratoire</strong> (EDA) approfondie</li>
                <li>🤖 <strong>Machine Learning :</strong> Clustering K-Means</li>
                <li>📈 <strong>Visualisations interactives</strong> avec Plotly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# KPI ROW - Indicateurs clés de performance
# ============================================
st.markdown('<h2 class="section-header">📊 Indicateurs clés de performance</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = df_filtered['sales'].sum()
    st.markdown(f"""
        <div class="kpi-card kpi-card-sales">
            <div class="kpi-title">💰 CHIFFRE D'AFFAIRES</div>
            <div class="kpi-value">${total_sales:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    total_profit = df_filtered['profit'].sum()
    profit_class = "kpi-card-profit" if total_profit >= 0 else "kpi-card-loss"
    st.markdown(f"""
        <div class="kpi-card {profit_class}">
            <div class="kpi-title">📈 PROFIT TOTAL</div>
            <div class="kpi-value">${total_profit:,.0f}</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    margin_color = COLORS['profit'] if margin >= 0 else COLORS['loss']
    st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, {margin_color} 0%, #333 100%);">
            <div class="kpi-title">📊 Pourcentage du profit</div>
            <div class="kpi-value">{margin:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    avg_discount = df_filtered['discount'].mean() * 100
    st.markdown(f"""
        <div class="kpi-card" style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);">
            <div class="kpi-title">🏷️ REMISE MOYENNE</div>
            <div class="kpi-value">{avg_discount:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================
# STATISTIQUES GLOBALES
# ============================================
st.markdown('<h2 class="section-header">📈 Vue d\'ensemble</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Top performances
    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;">
        <h3 style="color: #1f77b4; margin-top: 0; margin-bottom: 1.5rem; font-size: 1.4rem; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem;">
            🏆 Top Performances
        </h3>
    """, unsafe_allow_html=True)
    
    top_product = df_filtered.groupby('product_name')['sales'].sum().nlargest(1)
    top_category = df_filtered.groupby('category')['sales'].sum().nlargest(1)
    top_region = df_filtered.groupby('region')['sales'].sum().nlargest(1)
    
    st.markdown(f"""
        <div class="stat-card" style="border-left-color: #1f77b4;">
            <span style="color: #1f77b4; font-weight: bold;">🥇 Meilleur produit :</span> 
            <span style="color: #333333;">{top_product.index[0]}</span><br>
            <span style="color: #1f77b4; font-size: 1.2rem; font-weight: bold;">${top_product.values[0]:,.0f}</span>
        </div>
        <div class="stat-card" style="border-left-color: #1f77b4;">
            <span style="color: #1f77b4; font-weight: bold;">🥇 Meilleure catégorie :</span> 
            <span style="color: #333333;">{top_category.index[0]}</span><br>
            <span style="color: #1f77b4; font-size: 1.2rem; font-weight: bold;">${top_category.values[0]:,.0f}</span>
        </div>
        <div class="stat-card" style="border-left-color: #1f77b4;">
            <span style="color: #1f77b4; font-weight: bold;">🥇 Meilleure région :</span> 
            <span style="color: #333333;">{top_region.index[0]}</span><br>
            <span style="color: #1f77b4; font-size: 1.2rem; font-weight: bold;">${top_region.values[0]:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Points d'attention
    st.markdown("""
    <div style="background-color: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;">
        <h3 style="color: #d62728; margin-top: 0; margin-bottom: 1.5rem; font-size: 1.4rem; border-bottom: 2px solid #d62728; padding-bottom: 0.5rem;">
            ⚠️ Points d'attention
        </h3>
    """, unsafe_allow_html=True)
    
    loss_rate = (df_filtered['profit'] < 0).mean() * 100
    total_losses = abs(df_filtered[df_filtered['profit'] < 0]['profit'].sum())
    avg_loss = abs(df_filtered[df_filtered['profit'] < 0]['profit'].mean())
    
    st.markdown(f"""
        <div class="stat-card" style="border-left-color: #d62728;">
            <span style="color: #d62728; font-weight: bold;">📉 Taux de pertes :</span> 
            <span style="color: #333333; font-size: 1.2rem; font-weight: bold;">{loss_rate:.1f}%</span>
            <span style="color: #666666;"> des commandes</span>
        </div>
        <div class="stat-card" style="border-left-color: #d62728;">
            <span style="color: #d62728; font-weight: bold;">💰 Pertes totales :</span> 
            <span style="color: #333333; font-size: 1.2rem; font-weight: bold;">${total_losses:,.0f}</span>
        </div>
        <div class="stat-card" style="border-left-color: #d62728;">
            <span style="color: #d62728; font-weight: bold;">📊 Perte moyenne :</span> 
            <span style="color: #333333; font-size: 1.2rem; font-weight: bold;">${avg_loss:,.0f}</span>
            <span style="color: #666666;"> par commande</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<style>

/* ===== TAB STYLE GENERAL ===== */
button[data-baseweb="tab"] {
    font-size: 35px !important;
    font-weight: 800 !important;
    padding: 18px 32px !important;
    margin-right: 5px;
    border-radius: 10px 10px 0px 0px;
    color: #444;
    transition: all 0.3s ease;
}

/* ===== TAB HOVER EFFECT ===== */
button[data-baseweb="tab"]:hover {
    background-color: #f0f2f6;
    color: #000;
}

/* ===== TAB ACTIVE ===== */
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #ffffff;
    color: #000000;
    border-bottom: 4px solid #4CAF50;
    font-size: 42px !important;
}

/* ===== enlever la couleur rouge par défault ===== */
button[data-baseweb="tab"]::after {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# TABS PRINCIPAUX
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Analyse Temporelle",
    "🏷️ Analyse Produits",
    "🌍 Analyse Géographique",
    "🚚 Logistique",
    "👥 Clustering Global",
    "🔍 Analyse par Cluster",
    "📊 Aller plus loin"
])


# ============================================
# TAB 1: ANALYSE TEMPORELLE
# ============================================
with tab1:
    st.markdown('<h2 class="section-header">📈 Évolution des ventes et profits</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution trimestrielle
        quarterly = (
            df_filtered
            .groupby('year_quarter')[['sales', 'profit']]
            .sum()
            .reset_index()
        )
        
        fig = px.line(
            quarterly,
            x='year_quarter',
            y=['sales', 'profit'],
            title="Évolution trimestrielle",
            labels={'value': 'Montant ($)', 'variable': 'Métrique'},
            color_discrete_map={'sales': COLORS['sales'], 'profit': COLORS['profit']}
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode='x unified',
            template="plotly_white",
            font=dict(color="black")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit mensuel moyen
        monthly_profit = (
            df_filtered
            .groupby('month_name')['profit']
            .mean()
            .reindex(['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin',
                      'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'])
        )
        
        colors_month = [COLORS['profit'] if x >= 0 else COLORS['loss'] for x in monthly_profit.values]
        
        fig = px.bar(
            x=monthly_profit.index,
            y=monthly_profit.values,
            title="Profit moyen par mois",
            labels={'x': 'Mois', 'y': 'Profit moyen ($)'}
        )
        fig.update_traces(marker_color=colors_month)
        st.plotly_chart(fig, use_container_width=True)
    
    # Évolution annuelle
    st.markdown('<h3 class="subsection-header">📅 Performance annuelle</h3>', unsafe_allow_html=True)
    
    yearly = df_filtered.groupby('year')[['sales', 'profit']].sum().reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(x=yearly['year'], y=yearly['sales'], name="Ventes", 
               marker_color=COLORS['sales']),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=yearly['year'], y=yearly['profit'], name="Profit", 
                   mode='lines+markers', 
                   line=dict(color=COLORS['profit'], width=3),
                   marker=dict(size=10)),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Ventes annuelles vs Profit",
        hovermode='x unified',
        template="plotly_white",
        font=dict(color="black")
    )
    fig.update_yaxes(title_text="Ventes ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 2: ANALYSE PRODUITS
# ============================================
with tab2:
    st.markdown('<h2 class="section-header">🏷️ Analyse par catégorie</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance par catégorie
        cat_perf = (
            df_filtered
            .groupby('category')
            .agg({'sales': 'sum', 'profit': 'sum', 'discount': 'mean'})
            .reset_index()
        )
        cat_perf['margin'] = (cat_perf['profit'] / cat_perf['sales'] * 100).round(1)
        
        fig = px.bar(
            cat_perf,
            x='category',
            y=['sales', 'profit'],
            title="Ventes et Profits par catégorie",
            barmode='group',
            color_discrete_map={'sales': COLORS['sales'], 'profit': COLORS['profit']}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Impact du discount sur le profit
        discount_impact = (
            df_filtered
            .groupby('discount_bin')
            .agg({
                'profit': 'mean',
                'sales': 'count'
            })
            .reset_index()
        )
        
        fig = px.bar(
            discount_impact,
            x='discount_bin',
            y='profit',
            title="Profit moyen par niveau de discount",
            color='profit',
            color_continuous_scale=[COLORS['loss'], 'yellow', COLORS['profit']]
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS['loss'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Sous-catégories avec graphique miroir
    st.markdown('<h3 class="subsection-header">📦 Détail par sous-catégorie</h3>', unsafe_allow_html=True)
    
    subcat = (
        df_filtered
        .groupby('subcategory')[['sales', 'profit']]
        .sum()
        .sort_values('sales', ascending=True)
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=subcat.index,
        x=-subcat['sales'],
        name='Ventes',
        orientation='h',
        marker_color=COLORS['sales'],
        hovertemplate='Ventes: $%{x:.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        y=subcat.index,
        x=subcat['profit'],
        name='Profit',
        orientation='h',
        marker_color=COLORS['profit'],
        hovertemplate='Profit: $%{x:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Ventes vs Profits par Sous-Catégorie",
        barmode='overlay',
        height=600,
        hovermode='y unified'
    )
    
    fig.add_shape(
        type="line",
        x0=0, x1=0,
        y0=-0.5, y1=len(subcat)-0.5,
        line=dict(color=COLORS['loss'], width=2, dash="dash")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3: ANALYSE GÉOGRAPHIQUE
# ============================================
with tab3:
    st.markdown('<h2 class="section-header">🌍 Performance par région</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance par région
        region_perf = (
            df_filtered
            .groupby('region')
            .agg({
                'sales': 'sum',
                'profit': 'sum',
                'customer_id': 'nunique'
            })
            .reset_index()
        )
        region_perf['margin'] = (region_perf['profit'] / region_perf['sales'] * 100).round(1)
        
        colors_region = [COLORS['profit'] if x >= 0 else COLORS['loss'] for x in region_perf['profit']]
        
        fig = px.bar(
            region_perf,
            x='region',
            y='profit',
            title="Profit par région",
            color='profit',
            color_continuous_scale=[COLORS['loss'], 'lightgray', COLORS['profit']],
            text='margin'
        )
        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside',
            marker_color=colors_region
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS['loss'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Taux de perte par région
        loss_by_region = df_filtered.groupby('region')['is_loss'].mean().reset_index()
        loss_by_region['is_loss'] = loss_by_region['is_loss'] * 100
        
        fig = px.pie(
            loss_by_region,
            values='is_loss',
            names='region',
            title="Distribution des pertes par région",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 4: LOGISTIQUE
# ============================================
with tab4:
    st.markdown('<h2 class="section-header">🚚 Analyse logistique</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Délai moyen par mode
        shipping_by_mode = (
            df_filtered
            .groupby('ship_mode')['shipping_time_days']
            .mean()
            .reset_index()
        )
        
        couleurs_ship = {
            "Same Day": COLORS['loss'],
            "First Class": COLORS['sales'],
            "Second Class": "#ff9800",
            "Standard Class": COLORS['profit']
        }
        
        fig = px.bar(
            shipping_by_mode,
            x='ship_mode',
            y='shipping_time_days',
            title="Délai moyen par mode d'expédition",
            color='ship_mode',
            color_discrete_map=couleurs_ship
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit par mode
        profit_by_mode = (
            df_filtered
            .groupby('ship_mode')
            .agg({'profit': 'sum', 'sales': 'count'})
            .reset_index()
        )
        
        fig = px.bar(
            profit_by_mode,
            x='ship_mode',
            y='profit',
            title="Profit par mode d'expédition",
            color='ship_mode',
            color_discrete_map=couleurs_ship,
            text_auto='.2s'
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS['loss'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Corrélation temps de livraison / profit
    st.markdown('<h3 class="subsection-header">⏱️ Impact du délai de livraison</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        corr = df_filtered[['shipping_time_days', 'profit']].corr(method='spearman').iloc[0,1]
        st.metric("Corrélation (Spearman)", f"{corr:.3f}")
    
    with col2:
        avg_shipping = df_filtered['shipping_time_days'].mean()
        st.metric("Délai moyen (jours)", f"{avg_shipping:.1f}")

# ============================================
# TAB 5: CLUSTERING GLOBAL
# ============================================
with tab5:
    st.markdown('<h2 class="section-header">👥 Segmentation Client (K-Means)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🎯 Analyse des segments clients</h4>
        <p>
        Segmentation basée sur 4 dimensions : <span class="profit-text">profit total</span>, 
        <span class="sales-text">fréquence d'achat</span>, 
        <span class="sales-text">panier moyen</span> et 
        <span class="loss-text">remise moyenne</span>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="background: {COLORS['sales']};">
            <div class="kpi-title">👥 Clients analysés</div>
            <div class="kpi-value">{len(client_clean)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="background: {COLORS['profit']};">
            <div class="kpi-title">📊 Variance expliquée</div>
            <div class="kpi-value">{pca_ratio.sum()*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    
    st.markdown("---")
    
    # Visualisation 3D PCA
    st.markdown('<h3 class="subsection-header">🔮 Projection 3D des clusters (PCA)</h3>', unsafe_allow_html=True)

    client_clean['cluster'] = client_clean['cluster'].astype('category')

    fig_3d = px.scatter_3d(
        client_clean,
        x='pca_1',
        y='pca_2',
        z='pca_3',
        color='cluster',
        title="Visualisation des clusters dans l'espace réduit",
        opacity=0.6,
        hover_data=['total_profit', 'nb_orders', 'avg_basket'],
        color_discrete_map={
            0: "#5B00A5",
            1: "#B12A90", 
            2: "#F08A3C",
            3: "#F0F921"
        }
    )
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)


    # Profils des centroïdes
    st.markdown('<h3 class="subsection-header">📋 Profils des centroïdes</h3>', unsafe_allow_html=True)
    
    styled_profile = cluster_profile.style.format({
        'total_profit': '${:,.0f}',
        'nb_orders': '{:.1f}',
        'avg_basket': '${:,.0f}',
        'avg_discount': '{:.1%}',
        'nb_clients': '{:,.0f}',
    })
    
    st.dataframe(styled_profile, use_container_width=True)
    
    # Clients les plus proches des 4 clusters
    st.markdown("---")   
    st.markdown('<h3 class="subsection-header">🎯 Clients les plus proches des 4 clusters</h3>', unsafe_allow_html=True)

    clients_list = []

    for i in range(4):
        cluster_mask = client_clean["cluster"] == i
        X_cluster = X_clean[cluster_mask.values]
        centroid = kmeans.cluster_centers_[i]
        
        # Distance euclidienne
        distances = ((X_cluster - centroid) ** 2).sum(axis=1) ** 0.5
        closest_index = distances.argmin()
        
        closest_client = (
            client_clean[cluster_mask]
            .iloc[[closest_index]]
            .drop(columns=["cluster", "pca_1", "pca_2", "pca_3"], errors='ignore')
        )
        
        closest_client.index = [f"Client du Cluster {i}"]
        closest_client = (
            closest_client
            .reset_index()
            .rename(columns={"index": "client"})
        )
        
        clients_list.append(closest_client)

    clients_representatifs = pd.concat(clients_list)

    st.dataframe(
        clients_representatifs.style.format({
            "total_profit": "${:,.2f}",
            "avg_basket": "${:,.2f}",
            "avg_discount": "{:.2%}"
        }).hide(axis="index"),
        use_container_width=True
    )
    
    # Analyse des catégories par cluster
    st.markdown('<h3 class="subsection-header">🏷️ Répartition des catégories par cluster</h3>', unsafe_allow_html=True)
    
    category_pivot = (
        df_clusters
        .groupby(["cluster", "category"])
        .size()
        .reset_index(name="count")
        .pivot(index="cluster", columns="category", values="count")
        .fillna(0)
        .astype(int)
    )
    
    profit_category_pivot = (
        df_clusters
        .groupby(["cluster", "category"])["profit"]
        .sum()
        .reset_index()
        .pivot(index="cluster", columns="category", values="profit")
        .round(2)
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Volume par catégorie**")
        st.dataframe(category_pivot, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Profit par catégorie**")
        styled_profit_cat = profit_category_pivot.style.format('${:,.0f}')
        st.dataframe(styled_profit_cat, use_container_width=True)
    
    # Analyse des régions par cluster
    st.markdown('<h3 class="subsection-header">🌍 Répartition géographique par cluster</h3>', unsafe_allow_html=True)
    
    region_count_cluster = df_clusters.groupby(["cluster", "region"]).size().reset_index(name="count")
    profit_region_cluster = df_clusters.groupby(["cluster", "region"])["profit"].sum().reset_index()
    
    region_pivot = region_count_cluster.pivot(index="cluster", columns="region", values="count").fillna(0).astype(int)
    profit_region_pivot = profit_region_cluster.pivot(index="cluster", columns="region", values="profit").fillna(0).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Volume par région**")
        st.dataframe(region_pivot, use_container_width=True)
    
    with col2:
        st.markdown("**💰 Profit par région**")
        styled_profit_region = profit_region_pivot.style.format('${:,.0f}')
        st.dataframe(styled_profit_region, use_container_width=True)
    
    # Taux de perte par cluster
    st.markdown('<h3 class="subsection-header">⚠️ Taux de commandes en perte</h3>', unsafe_allow_html=True)
    
    loss_rate_cluster = (
        df_clusters
        .assign(loss=df_clusters["profit"] < 0)
        .groupby("cluster")["loss"]
        .mean()
        .round(4)
    )
    
    loss_df = pd.DataFrame({
        'cluster': loss_rate_cluster.index,
        'loss_rate': loss_rate_cluster.values * 100
    })
    
    fig_loss_rate = px.bar(
        loss_df,
        x='cluster',
        y='loss_rate',
        title="Taux de perte par cluster",
        color='loss_rate',
        color_continuous_scale=['#2ca02c', '#ffffbf', '#d62728'],
        text=loss_df['loss_rate'].round(1).astype(str) + '%'
    )
    fig_loss_rate.update_traces(textposition='outside')
    fig_loss_rate.update_layout(height=400)
    st.plotly_chart(fig_loss_rate, use_container_width=True)

# ============================================
# TAB 6: ANALYSE PAR CLUSTER
# ============================================
with tab6:
    st.markdown('<h2 class="section-header">🔍 Exploration Détaillée par Cluster</h2>', unsafe_allow_html=True)
    
    # Sélection du cluster
    selected_cluster = st.selectbox(
        "Choisissez un cluster à analyser en profondeur",
        options=sorted(client_clean['cluster'].unique()),
        format_func=lambda x: f"Cluster {x} ({cluster_profile.loc[x, 'nb_clients']} clients)"
    )
    
    # Récupération des données du cluster
    clients_cluster = client_clean[client_clean["cluster"] == selected_cluster].index
    df_cluster = df_clusters[df_clusters['customer_id'].isin(clients_cluster)]
    df_client_cluster = client_clean[client_clean["cluster"] == selected_cluster]
    
    # KPIs du cluster
    st.markdown(f"<h3 class='subsection-header'>📊 Statistiques du Cluster {selected_cluster}</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de clients", len(clients_cluster))
    with col2:
        st.metric("Profit total", f"${df_cluster['profit'].sum():,.0f}")
    with col3:
        st.metric("Panier moyen", f"${df_cluster['sales'].mean():,.0f}")
    with col4:
        loss_rate = (df_cluster['profit'] < 0).mean() * 100
        st.metric("Taux de perte", f"{loss_rate:.1f}%")
    
    # Distributions
    st.markdown(f"<h3 class='subsection-header'>📈 Distributions - Cluster {selected_cluster}</h3>", unsafe_allow_html=True)
    fig_dist = plot_distribution_cluster(df_client_cluster, selected_cluster)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Graphique miroir
    st.markdown(f"<h3 class='subsection-header'>📊 Ventes vs Profits par Sous-Catégorie</h3>", unsafe_allow_html=True)
    fig_mirror = plot_mirror_chart(df_cluster, selected_cluster)
    st.plotly_chart(fig_mirror, use_container_width=True)
    
    # Catégories dominantes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h4>🏷️ Catégories - Cluster {selected_cluster}</h4>", unsafe_allow_html=True)
        cat_counts = df_cluster['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        
        modern_colors = [COLORS['sales'], COLORS['profit'], COLORS['loss']]
        
        fig_cat = px.bar(
            cat_counts,
            x='category',
            y='count',
            color='category',
            text='count',
            title=f"Catégories dominantes",
            color_discrete_sequence=modern_colors
        )
        fig_cat.update_traces(textposition="outside", marker_line_width=1.5)
        fig_cat.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        st.markdown(f"<h4>🌍 Répartition géographique</h4>", unsafe_allow_html=True)
        region_counts = df_cluster['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        
        gradient_green = ["#e5f5e0", "#a1d99b", "#31a354", "#006d2c"]
        
        fig_region = px.pie(
            region_counts,
            values='count',
            names='region',
            hole=0.65,
            color='region',
            color_discrete_sequence=gradient_green
        )
        fig_region.update_traces(textinfo='percent', marker=dict(line=dict(color='white', width=2)))
        fig_region.update_layout(height=400)
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Radar chart
    st.markdown(f"<h3 class='subsection-header'>🕸️ Profil Radar Normalisé</h3>", unsafe_allow_html=True)
    fig_radar = plot_radar_profile(cluster_profile, selected_cluster)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Carte des États
    st.markdown(f"<h3 class='subsection-header'>🗺️ Performance par État</h3>", unsafe_allow_html=True)
    fig_map = plot_state_map(df_cluster, selected_cluster)
    if fig_map:
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("⚠️ Pas assez de données pour afficher la carte des États pour ce cluster.")
    
    # Top et Bottom États
    state_perf = (
        df_cluster
        .groupby('state')
        .agg({'profit': 'sum', 'sales': 'sum', 'order_id': 'nunique'})
        .reset_index()
        .sort_values('profit', ascending=False)
    )
    
    if not state_perf.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏆 Top 5 États (Profit)**")
            top_states = state_perf.head(5)[['state', 'profit', 'sales', 'order_id']]
            top_states['profit'] = top_states['profit'].apply(lambda x: f"${x:,.0f}")
            top_states['sales'] = top_states['sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_states, use_container_width=True)
        
        with col2:
            st.markdown(f"**📉 Bottom 5 États (Profit)**")
            bottom_states = state_perf.tail(5)[['state', 'profit', 'sales', 'order_id']]
            bottom_states['profit'] = bottom_states['profit'].apply(lambda x: f"${x:,.0f}")
            bottom_states['sales'] = bottom_states['sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(bottom_states, use_container_width=True)

# ============================================
# TAB 7: ALLER PLUS LOIN
# ============================================
with tab7:
    st.markdown('<h2 class="section-header">📊 Analyse Croisée Avancée</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🔬 Analyse approfondie des sous-catégories</h4>
        <p>
        Explorez les interactions entre les clusters et les sous-catégories de produits pour identifier 
        des opportunités d'optimisation et des patterns de consommation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-header">Analyse détaillée des sous-catégories par cluster</h3>', unsafe_allow_html=True)
    
    fig_volume, fig_profit, fig_loss = plot_subcategory_analysis(df_clusters)
    
    tab_vol, tab_prof, tab_los = st.tabs(["📦 Volume", "💰 Profit", "⚠️ Taux de perte"])
    
    with tab_vol:
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab_prof:
        st.plotly_chart(fig_profit, use_container_width=True)
    
    with tab_los:
        st.plotly_chart(fig_loss, use_container_width=True)
    

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer" style="color:white;">
    <p>📊 Superstore Advanced Analytics Dashboard | Version enrichie avec segmentation avancée</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">
        Données : Sample Superstore | 
        <span class="profit-text">■ Profit</span> | 
        <span class="loss-text">■ Perte</span> | 
        <span class="sales-text">■ Ventes</span>
    </p>
    <p style="font-size: 0.8rem; opacity: 0.5;">© 2024 - Analyse complète avec K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)