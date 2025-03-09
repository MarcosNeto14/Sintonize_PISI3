import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
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

# Normalizar os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

# Clusterização
st.title("🔗 Clusterização de Músicas")
st.markdown("""
A análise a seguir utiliza **K-Means** para agrupar músicas com base em seus atributos acústicos.
""")

# Método do Cotovelo
st.write("### Método do Cotovelo")
inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(range(2, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
st.pyplot(fig)

# Definir número de clusters
num_clusters = st.slider("Selecione o número de clusters:", 2, 10, 4)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

# Gráfico de Silhueta
st.write("### Análise de Silhueta")
silhouette_avg = silhouette_score(scaled_data, df_filtered['cluster'])
st.write(f"Média do Coeficiente de Silhueta: {silhouette_avg:.2f}")

silhouette_values = silhouette_samples(scaled_data, df_filtered['cluster'])
y_lower, y_upper = 0, 0
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("Set1", num_clusters)

for i in range(num_clusters):
    cluster_silhouette_values = silhouette_values[df_filtered['cluster'] == i]
    cluster_silhouette_values.sort()
    y_upper += len(cluster_silhouette_values)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values, facecolor=colors[i], edgecolor='black')
    ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_values), str(i))
    y_lower = y_upper

ax.axvline(x=silhouette_avg, color="red", linestyle="--")
ax.set_xlabel("Coeficiente de Silhueta")
ax.set_ylabel("Clusters")
ax.set_yticks([])
st.pyplot(fig)

# Gráfico de Dispersão dos Clusters
st.write("### Visualização dos Clusters")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title(f"Clusters de Músicas por Danceability e Energy ({num_clusters} clusters)")
st.pyplot(fig)

# Interpretação dos Clusters
st.write("#### Interpretação dos Clusters")
st.markdown("""
- **Cluster 1**: Músicas com alta energia e baixa dançabilidade (ex.: rock dos anos 1970).
- **Cluster 2**: Músicas com alta dançabilidade e energia moderada (ex.: pop dos anos 1980).
- **Cluster 3**: Músicas acústicas e melancólicas (ex.: folk dos anos 1960).
- **Cluster 4**: Músicas com baixa energia e alta dançabilidade (ex.: jazz suave).
""")

# Explicação da Importância da Análise
st.markdown("""
### 📊 Importância da Análise de Silhueta

O gráfico de silhueta nos permite avaliar a qualidade da clusterização. Quanto maior o coeficiente de silhueta, melhor definido está o cluster. A linha vermelha indica a média geral, e valores negativos sugerem que algumas músicas podem estar em clusters inadequados.

Além disso, a visualização dos clusters nos permite identificar padrões e características comuns entre músicas agrupadas.
""")
