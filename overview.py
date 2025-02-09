import streamlit as st
import pandas as pd
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

# Visão geral
st.title("📊 Análise da Evolução Musical ao Longo das Décadas")
st.subheader("Dataset: Music Dataset : 1950 to 2019")
st.subheader("Usabilidade: 9.41")

st.markdown("""
O Sintonize utiliza um dataset contendo informações sobre músicas de diversas décadas, incluindo atributos como dançabilidade,
energia, valência emocional e tristeza. A análise busca identificar padrões e tendências históricas, considerando eventos culturais
e tecnológicos, como o impacto da internet nos gêneros musicais.
""")

# Métricas gerais
df['year'] = df['year'].astype('Int64')
col1, col2, col3 = st.columns(3)
col1.metric("🎵 Total de Músicas", f"{len(df):,}".replace(",", "."))
col2.metric("🗓️ Período Coberto", f"{int(df['year'].min())} a {int(df['year'].max())}")
col3.metric("🎼 Gêneros Únicos", f"{df['genre'].nunique()}")

# Amostra dos dados
st.write("### Amostra dos Dados")
st.dataframe(df.head(10))

# Estatísticas descritivas
total_genres = df['genre'].nunique()
st.write(f"#### Estatísticas Descritivas do Dataset ({total_genres} gêneros musicais)")
st.dataframe(df.describe())
