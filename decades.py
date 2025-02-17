import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Filtro de intervalo de anos
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

# Filtro de gêneros musicais
if 'selected_genres' not in st.session_state:
    st.session_state.selected_genres = sorted(df['genre'].dropna().unique())

selected_genres = st.multiselect(
    "Selecione os gêneros musicais para análise:",
    options=sorted(df['genre'].dropna().unique()),
    default=st.session_state.selected_genres
)

st.session_state.selected_genres = selected_genres

# Filtragem de dados
filtered_df = df[
    (df['year'] >= min_year) & 
    (df['year'] <= max_year) & 
    (df['genre'].isin(selected_genres))
]

filtered_df['decade'] = (filtered_df['year'] // 10 * 10).astype(int)

# Distribuição de gêneros por década
genre_distribution = filtered_df.groupby(['decade', 'genre']).size().unstack(fill_value=0)

# Exibição do Heatmap
st.write("### Heatmap de Gêneros por Década")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(genre_distribution, annot=False, cmap="coolwarm", ax=ax)
plt.xticks(rotation=0, ha='center')
ax.set_title(f"Distribuição de Gêneros por Década ({min_year}-{max_year})")
st.pyplot(fig)
