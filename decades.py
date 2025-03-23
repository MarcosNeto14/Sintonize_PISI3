import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
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

# Página de Distribuição por Décadas
st.title("📊 Distribuição de Músicas por Décadas")
st.markdown("""
Os gráficos mostram a quantidade de registros por gênero em cada década, ajudando a identificar as tendências musicais.
""")

# =====================================
# Aplicando One-Hot Encoding para visualização
# =====================================
st.write("### 🔧 Aplicando One-Hot Encoding para Visualização")
st.markdown("""
O **One-Hot Encoding** é aplicado na coluna `genre` para criar uma matriz binária que pode ser usada em visualizações, como heatmaps.
""")

# Remover linhas com valores nulos na coluna 'genre'
df = df.dropna(subset=['genre'])

# Aplicando o One-Hot Encoding
encoder = OneHotEncoder(drop='if_binary')  # drop='if_binary' para evitar multicolinearidade
genre_encoded = encoder.fit_transform(df[['genre']])  # Garantir que a entrada seja 2D

# Criando um DataFrame com as colunas codificadas
encoded_columns = encoder.get_feature_names_out(['genre'])
df_encoded = pd.DataFrame(genre_encoded.toarray(), columns=encoded_columns)  # Usar .toarray() para converter para matriz densa

# Concatenando o DataFrame codificado com o DataFrame original
df = pd.concat([df, df_encoded], axis=1)

# Exibindo uma amostra aleatória do DataFrame codificado
st.write("#### Amostra Aleatória do DataFrame com One-Hot Encoding Aplicado")
st.write(df_encoded.sample(10))  # Exibe 10 linhas aleatórias

# =====================================
# Filtro de intervalo de anos
# =====================================
if 'min_year' not in st.session_state:
    st.session_state.min_year = int(df['year'].min())
if 'max_year' not in st.session_state:
    st.session_state.max_year = int(df['year'].max())

min_year, max_year = st.slider(
    "Selecione o intervalo de anos para análise:",
    int(df['year'].min()), int(df['year'].max()),
    (st.session_state.min_year, st.session_state.max_year)
)

st.session_state.min_year = min_year
st.session_state.max_year = max_year

# =====================================
# Filtro de gêneros musicais
# =====================================
if 'selected_genres' not in st.session_state:
    st.session_state.selected_genres = sorted(df['genre'].dropna().unique())

selected_genres = st.multiselect(
    "Selecione os gêneros musicais para análise:",
    options=sorted(df['genre'].dropna().unique()),
    default=st.session_state.selected_genres
)

st.session_state.selected_genres = selected_genres

# =====================================
# Filtragem de dados
# =====================================
filtered_df = df[
    (df['year'] >= min_year) & 
    (df['year'] <= max_year) & 
    (df['genre'].isin(selected_genres))
]

filtered_df['decade'] = (filtered_df['year'] // 10 * 10).astype(int)

# =====================================
# Distribuição de gêneros por década (usando colunas codificadas)
# =====================================
st.write("### 📊 Distribuição de Gêneros por Década (One-Hot Encoding)")
st.markdown("""
O heatmap abaixo mostra a distribuição de gêneros musicais por década, utilizando as colunas codificadas.
""")

# Agrupar por década e somar as colunas codificadas
genre_distribution = filtered_df.groupby('decade')[encoded_columns].sum()

# Exibição do Heatmap
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(genre_distribution.T, annot=False, cmap="coolwarm", ax=ax)  # Transpor para melhor visualização
plt.xticks(rotation=0, ha='center')
ax.set_title(f"Distribuição de Gêneros por Década ({min_year}-{max_year})")
ax.set_xlabel("Década")
ax.set_ylabel("Gênero Musical")
st.pyplot(fig)