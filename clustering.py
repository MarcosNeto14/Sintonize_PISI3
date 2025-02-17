import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import load_data

# Caminho do dataset
parquet_path = "dataset/parquet/tcc_ceds_music.parquet"
df = load_data(parquet_path)
if df is None:
    st.stop()

# Verificar se a coluna 'year' existe
if 'year' not in df.columns:
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
        df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
        df['decade'] = (df['year'] // 10 * 10).astype('Int64')
    else:
        st.error("Erro: A coluna 'year' não foi encontrada no dataset. Verifique o formato do arquivo.")
        st.stop()

# Garantir que os atributos sejam numéricos
acoustic_features = ['danceability', 'energy', 'acousticness', 'sadness', 'instrumentalness']
df[acoustic_features] = df[acoustic_features].apply(pd.to_numeric, errors='coerce')
df_filtered = df.dropna(subset=acoustic_features)

# Clusterização
st.title("🔗 Clusterização de Músicas")
st.markdown("""
A análise a seguir utiliza **K-Means** para agrupar músicas com base em seus atributos acústicos.
""")

# Normalizar os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

# Definir número de clusters
num_clusters = st.slider("Selecione o número de clusters:", 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

# Gráfico de dispersão por Danceability e Energy
st.write("### Clusterização de Músicas com Base em Danceability e Energy")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title("Clusters de Músicas por Danceability e Energy")
st.pyplot(fig)

# Nova análise: Clusterização com 'sadness' e 'energy'
st.write("### Clusterização com Base em Sadness e Energy")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=df_filtered['sadness'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set2")
plt.xlabel("Sadness")
plt.ylabel("Energy")
plt.title("Clusters de Músicas por Sadness e Energy")
st.pyplot(fig)
