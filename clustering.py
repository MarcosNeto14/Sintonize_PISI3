import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import load_data

parquet_path = "dataset/parquet/tcc_ceds_music.parquet"
df = load_data(parquet_path)

if df is None:
    st.stop()

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

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

st.title("🔗 Clusterização de Músicas")
st.markdown("""
Esta aplicação utiliza **K-Means** para agrupar músicas com base em seus atributos acústicos. 
Abaixo, você pode simular a classificação de uma música em um cluster inserindo os valores dos atributos.
""")


st.write("### 🎵 Simulação de Previsão de Cluster")
st.markdown("""
Insira os valores dos atributos acústicos para simular a classificação da música em um cluster.
""")

# Inputs para os atributos acústicos
col1, col2 = st.columns(2)
with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
with col2:
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
    sadness = st.slider("Sadness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.5)

user_input = pd.DataFrame({
    'danceability': [danceability],
    'energy': [energy],
    'acousticness': [acousticness],
    'sadness': [sadness],
    'instrumentalness': [instrumentalness]
})


user_input_scaled = scaler.transform(user_input)

# Treinar o modelo K-Means com o conjunto de dados completo
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(scaled_data)  


predicted_cluster = kmeans.predict(user_input_scaled)


st.success(f"### 🎶 A música pertenceria ao **Cluster {predicted_cluster[0]}**")
st.markdown(f"""
Com base nos valores inseridos, a música foi classificada no **Cluster {predicted_cluster[0]}**.
""")

st.write("### 📊 Visualização dos Clusters")

num_clusters = st.slider("Selecione o número de clusters para visualização:", 2, 10, 4)

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

st.write("#### Clusterização de Músicas com Base em Danceability e Energy")
fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
plt.xlabel("Danceability")
plt.ylabel("Energy")
plt.title(f"Clusters de Músicas por Danceability e Energy ({num_clusters} clusters)")
st.pyplot(fig)

st.write("#### Interpretação dos Clusters")
st.markdown("""
- **Cluster 1**: Músicas com alta energia e baixa dançabilidade (ex.: rock dos anos 1970).
- **Cluster 2**: Músicas com alta dançabilidade e energia moderada (ex.: pop dos anos 1980).
- **Cluster 3**: Músicas acústicas e melancólicas (ex.: folk dos anos 1960).
- **Cluster 4**: Músicas com baixa energia e alta dançabilidade (ex.: jazz suave).
""")


st.write("### 📉 Método do Cotovelo")
st.markdown("""
O método do cotovelo ajuda a escolher o número ideal de clusters. 
O ponto onde a inércia começa a diminuir mais lentamente (o "cotovelo") indica o número ideal de clusters.
""")

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
st.pyplot(fig)