# superstore_dashboard/app.py

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
# CHARGEMENT DES DONNÉES (avec cache)
# ============================================
@st.cache_data
def load_and_clean_data():
    """
    Charge et nettoie les données du Superstore
    
    Returns:
        df: DataFrame nettoyé avec features engineering
    """
    
    with st.spinner("📥 Chargement des données..."):
        # Téléchargement du dataset depuis Kaggle
        path = kagglehub.dataset_download("vivek468/superstore-dataset-final")
        csv_file = os.path.join(path, "Sample - Superstore.csv")
        
        # Détection automatique de l'encodage
        with open(csv_file, "rb") as f:
            raw = f.read()
        enc = chardet.detect(raw)
        
        # Chargement du fichier CSV
        df = pd.read_csv(csv_file, encoding=enc['encoding'])
        
        # ===== NETTOYAGE DES DONNÉES =====
        # Nettoyage des noms de colonnes (minuscules, underscores)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        
        # Suppression de la colonne row_id si elle existe
        if "row_id" in df.columns:
            df = df.drop(columns=["row_id"])
        
        # Conversion des types de données
        df['postal_code'] = df['postal_code'].astype(str)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
        df['sales'] = df['sales'].astype(float)
        df['discount'] = df['discount'].astype(float)
        
        # Suppression des doublons
        df = df.drop_duplicates()
        
        # ===== FEATURE ENGINEERING =====
        # Calcul du temps de livraison en jours
        df['shipping_time_days'] = (df["ship_date"] - df['order_date']).dt.days
        
        # Extraction des composantes temporelles
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['quarter'] = df['order_date'].dt.quarter
        df['year_quarter'] = df['year'].astype(str) + ' Q' + df['quarter'].astype(str)
        
        # Noms des mois en français
        mois_fr = {
            1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
            5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
            9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
        }
        df['month_name'] = df['month'].map(mois_fr)
        
        # Catégories de discount
        df['discount_bin'] = pd.cut(
            df['discount'],
            bins=[-0.01, 0, 0.1, 0.2, 0.3, 1],
            labels=['0%', '0–10%', '10–20%', '20–30%', '30%+']
        )
        
        # Indicateur de perte (profit négatif)
        df['is_loss'] = df['profit'] < 0
        
        return df

# ============================================
# PRÉPARATION DES DONNÉES POUR LE CLUSTERING
# ============================================
@st.cache_data
def prepare_clustering_data(df):
    """
    Agrège les données par client pour le clustering
    
    Args:
        df: DataFrame original
        
    Returns:
        client_features: DataFrame avec les features par client
    """
    
    # Agrégation par client
    client_features = (
        df.groupby("customer_id")
        .agg({
            "profit": "sum",        # Profit total par client
            "order_id": "nunique",   # Nombre de commandes
            "sales": "mean",         # Panier moyen
            "discount": "mean"       # Discount moyen obtenu
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

# ============================================
# CLUSTERING AVEC K-MEANS
# ============================================
@st.cache_data
def perform_clustering(client_features, n_clusters=4):
    """
    Réalise le clustering K-Means sur les clients
    
    Args:
        client_features: DataFrame avec les features clients
        n_clusters: Nombre de clusters
        
    Returns:
        client_clean: DataFrame avec les clusters
        client_with_outliers: DataFrame avec indicateurs d'outliers
        pca_ratio: Variance expliquée par la PCA
    """
    
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    # Scaling robuste (moins sensible aux outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(client_features)
    
    # Détection des outliers avec Isolation Forest
    iso = IsolationForest(contamination=0.03, random_state=42)
    outliers = iso.fit_predict(X_scaled)
    client_features_with_outliers = client_features.copy()
    client_features_with_outliers["outlier"] = outliers
    
    # Données propres (sans outliers)
    X_clean = X_scaled[outliers == 1]
    client_clean = client_features[outliers == 1].copy()
    
    # Clustering avec K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clean)
    client_clean["cluster"] = clusters
    
    # PCA pour visualisation 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_clean)
    client_clean["pca_1"] = X_pca[:, 0]
    client_clean["pca_2"] = X_pca[:, 1]
    client_clean["pca_3"] = X_pca[:, 2]
    
    return client_clean, client_features_with_outliers, pca.explained_variance_ratio_

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
df = load_and_clean_data()
client_features = prepare_clustering_data(df)
client_clean, client_with_outliers, pca_ratio = perform_clustering(client_features, n_clusters=4)

# ============================================
# SIDEBAR - FILTRES
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shop.png", width=80)
    st.title("Superstore Dashboard")
    st.markdown("---")
    
    # Filtres principaux
    st.header("🔍 Filtres")
    
    # Filtre année
    available_years = sorted(df['year'].unique())
    selected_years = st.multiselect(
        "📅 Années",
        options=available_years,
        default=available_years
    )
    
    # Filtre région
    available_regions = sorted(df['region'].unique())
    selected_regions = st.multiselect(
        "🌍 Régions",
        options=available_regions,
        default=available_regions
    )
    
    # Filtre catégorie
    available_categories = sorted(df['category'].unique())
    selected_categories = st.multiselect(
        "🏷️ Catégories",
        options=available_categories,
        default=available_categories
    )
    
    # Filtre segment client
    available_segments = sorted(df['segment'].unique())
    selected_segments = st.multiselect(
        "👥 Segment client",
        options=available_segments,
        default=available_segments
    )
    
    st.markdown("---")
    
    # Statistiques rapides
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
st.markdown('<h1 class="main-header">📊 Superstore Analytics Dashboard</h1>', unsafe_allow_html=True)

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
            <div class="kpi-title">📊 Pourcentage de profit </div>
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

# ============================================
# TABS PRINCIPAUX
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Analyse Temporelle",
    "🏷️ Analyse Produits",
    "🌍 Analyse Géographique",
    "🚚 Logistique",
    "👥 Segmentation Clients"    
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
# TAB 5: SEGMENTATION CLIENTS (AVEC ANALYSE PAR ÉTAT)
# ============================================
with tab5:
    st.markdown('<h2 class="section-header">👥 Segmentation Client (K-Means Clustering)</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🎯 Objectif de la segmentation</h4>
        <p>
        Regrouper les clients en 4 segments homogènes basés sur leur comportement d'achat :
        <span class="sales-text">profit total</span>, 
        <span class="sales-text">fréquence d'achat</span>, 
        <span class="sales-text">panier moyen</span> et 
        <span class="sales-text">discount moyen</span>.
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
    
    # Visualisation 3D
    st.markdown('<h3 class="subsection-header">🔮 Visualisation 3D des clusters (PCA)</h3>', unsafe_allow_html=True)
    
    fig_3d = px.scatter_3d(
        client_clean,
        x='pca_1',
        y='pca_2',
        z='pca_3',
        color='cluster',
        title="Projection des clusters dans l'espace PCA",
        opacity=0.7,
        color_continuous_scale='viridis'
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Profil des clusters
    st.markdown('<h3 class="subsection-header">📊 Profil des clusters</h3>', unsafe_allow_html=True)
    
    cluster_profile = (
        client_clean
        .groupby('cluster')
        .agg({
            'total_profit': 'mean',
            'nb_orders': 'mean',
            'avg_basket': 'mean',
            'avg_discount': 'mean'
        })
        .round(2)
    )
    
    cluster_profile['nb_clients'] = client_clean.groupby('cluster').size()
    cluster_profile['pct_clients'] = (cluster_profile['nb_clients'] / len(client_clean) * 100).round(1)
    
    # Formatage du DataFrame pour l'affichage
    styled_df = cluster_profile.style.format({
        'total_profit': '${:,.0f}',
        'nb_orders': '{:.1f}',
        'avg_basket': '${:,.0f}',
        'avg_discount': '{:.1%}',
        'pct_clients': '{:.1f}%'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Radar chart
    st.markdown('<h3 class="subsection-header">🕸️ Profil radar des clusters</h3>', unsafe_allow_html=True)
    
    from sklearn.preprocessing import MinMaxScaler
    
    features_radar = ['total_profit', 'nb_orders', 'avg_basket', 'avg_discount']
    cluster_means = cluster_profile[features_radar]
    
    scaler = MinMaxScaler()
    cluster_scaled = scaler.fit_transform(cluster_means)
    
    fig_radar = go.Figure()
    
    colors_radar = px.colors.qualitative.Set1
    
    for i in range(len(cluster_scaled)):
        values = cluster_scaled[i].tolist()
        values += values[:1]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=features_radar + [features_radar[0]],
            fill='toself',
            name=f'Cluster {i}',
            line_color=colors_radar[i]
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Profils normalisés des clusters",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # ============================================
    # ANALYSE DÉTAILLÉE D'UN CLUSTER
    # ============================================
    st.markdown('<h3 class="subsection-header">🔍 Analyse détaillée par cluster</h3>', unsafe_allow_html=True)
    
    selected_cluster = st.selectbox(
        "Choisissez un cluster à analyser",
        options=sorted(client_clean['cluster'].unique())
    )
    
    # Récupérer les clients du cluster sélectionné
    clients_cluster = client_clean[client_clean['cluster'] == selected_cluster].index
    df_cluster = df[df['customer_id'].isin(clients_cluster)]
    
    # Statistiques du cluster
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre de clients", len(clients_cluster))
    with col2:
        st.metric("Profit total", f"${df_cluster['profit'].sum():,.0f}")
    with col3:
        st.metric("Panier moyen", f"${df_cluster['sales'].mean():,.0f}")
    with col4:
        st.metric("Taux de perte", f"{(df_cluster['profit'] < 0).mean()*100:.1f}%")
    
    # Première ligne de graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Catégories dominantes
        cat_counts = df_cluster['category'].value_counts().reset_index()
        cat_counts.columns = ['category', 'count']
        
        fig_cat = px.bar(
            cat_counts,
            x='category',
            y='count',
            title=f"🏷️ Catégories - Cluster {selected_cluster}",
            color='category',
            color_discrete_sequence=[COLORS['sales'], COLORS['profit'], COLORS['loss']]
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col2:
        # Répartition géographique (régions)
        region_counts = df_cluster['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        
        fig_region = px.pie(
            region_counts,
            values='count',
            names='region',
            title=f"🌍 Répartition par région - Cluster {selected_cluster}",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    # ============================================
    # PERFORMANCE PAR ÉTAT (SPÉCIFIQUE AU CLUSTER)
    # ============================================
    st.markdown(f"<h4>🗺️ Performance par État - Cluster {selected_cluster}</h4>", unsafe_allow_html=True)
    
    # Dictionnaire de conversion État → Code USPS
    us_state_abbrev = {
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
    
    # Agrégation par État pour le cluster
    state_perf_cluster = (
        df_cluster
        .groupby('state')
        .agg({
            'profit': 'sum',
            'sales': 'sum',
            'customer_id': 'nunique',
            'order_id': 'nunique'
        })
        .reset_index()
    )
    
    state_perf_cluster['state_code'] = state_perf_cluster['state'].map(us_state_abbrev)
    state_perf_cluster = state_perf_cluster.dropna(subset=['state_code'])
    state_perf_cluster = state_perf_cluster.sort_values('profit', ascending=False)
    
    if not state_perf_cluster.empty:
        # Carte choroplèthe
        fig_map = px.choropleth(
            state_perf_cluster,
            locations='state_code',
            locationmode='USA-states',
            color='profit',
            scope='usa',
            title=f"Profit par État - Cluster {selected_cluster}",
            color_continuous_scale=[COLORS['loss'], 'lightgray', COLORS['profit']],
            hover_data={
                'state': True,
                'profit': ':,.0f',
                'sales': ':,.0f',
                'customer_id': True,
                'order_id': True
            }
        )
        
        fig_map.update_layout(
            height=500,
            coloraxis_colorbar_title="Profit ($)"
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Top 5 États par profit
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏆 Top 5 États (Profit) - Cluster {selected_cluster}**")
            top_states = state_perf_cluster.head(5)[['state', 'profit', 'sales', 'order_id']]
            top_states['profit'] = top_states['profit'].apply(lambda x: f"${x:,.0f}")
            top_states['sales'] = top_states['sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(top_states, use_container_width=True)
        
        with col2:
            st.markdown(f"**📉 Bottom 5 États (Profit) - Cluster {selected_cluster}**")
            bottom_states = state_perf_cluster.tail(5)[['state', 'profit', 'sales', 'order_id']]
            bottom_states['profit'] = bottom_states['profit'].apply(lambda x: f"${x:,.0f}")
            bottom_states['sales'] = bottom_states['sales'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(bottom_states, use_container_width=True)
    else:
        st.warning("⚠️ Pas assez de données pour afficher la carte des États pour ce cluster.")
    
    # Deuxième ligne de graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance par sous-catégorie
        st.markdown(f"<h4>📦 Performance produits - Cluster {selected_cluster}</h4>", unsafe_allow_html=True)
        
        subcat_cluster = (
            df_cluster
            .groupby('subcategory')[['sales', 'profit']]
            .sum()
            .sort_values('sales')
        )
        
        fig_miroir = go.Figure()
        
        fig_miroir.add_trace(go.Bar(
            y=subcat_cluster.index,
            x=-subcat_cluster['sales'],
            name='Ventes',
            orientation='h',
            marker_color=COLORS['sales']
        ))
        
        fig_miroir.add_trace(go.Bar(
            y=subcat_cluster.index,
            x=subcat_cluster['profit'],
            name='Profit',
            orientation='h',
            marker_color=COLORS['profit']
        ))
        
        fig_miroir.update_layout(
            title=f"Ventes vs Profits par Sous-Catégorie",
            barmode='overlay',
            height=500,
            hovermode='y unified'
        )
        
        fig_miroir.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-0.5, y1=len(subcat_cluster)-0.5,
            line=dict(color=COLORS['loss'], width=2)
        )
        
        st.plotly_chart(fig_miroir, use_container_width=True)
    
    with col2:
        # Analyse des discounts
        st.markdown(f"<h4>🏷️ Impact des discounts - Cluster {selected_cluster}</h4>", unsafe_allow_html=True)
        
        discount_impact_cluster = (
            df_cluster
            .groupby('discount_bin')
            .agg({
                'profit': 'mean',
                'sales': 'count'
            })
            .reset_index()
        )
        
        fig_discount = px.bar(
            discount_impact_cluster,
            x='discount_bin',
            y='profit',
            title="Profit moyen par niveau de discount",
            color='profit',
            color_continuous_scale=[COLORS['loss'], 'yellow', COLORS['profit']]
        )
        fig_discount.add_hline(y=0, line_dash="dash", line_color=COLORS['loss'])
        st.plotly_chart(fig_discount, use_container_width=True)
    
    # Taux de perte comparatif
    loss_rate_cluster = (df_cluster['profit'] < 0).mean() * 100
    loss_rate_global = (df['profit'] < 0).mean() * 100
    delta = loss_rate_cluster - loss_rate_global
    
    st.metric(
        "📉 Taux de commandes en perte",
        f"{loss_rate_cluster:.1f}%",
        delta=f"{delta:+.1f}% vs global",
        delta_color="inverse"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("""
<div class="footer">
    <p>📊 Superstore Analytics Dashboard | Développé avec Streamlit</p>
    <p style="font-size: 0.9rem; opacity: 0.7;">
        Données : Sample Superstore | 
        <span class="profit-text">■ Profit</span> | 
        <span class="loss-text">■ Perte</span> | 
        <span class="sales-text">■ Ventes</span>
    </p>
    <p style="font-size: 0.8rem; opacity: 0.5;">© 2024 - Tous droits réservés</p>
</div>
""", unsafe_allow_html=True)