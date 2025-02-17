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

# Garantir que os atributos sejam numéricos
attributes = ["danceability", "acousticness", "energy", "valence", "loudness", "instrumentalness"]
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

# Página da Evolução Acústica
st.title("🎶 Evolução Acústica")
st.markdown("""
A linha do tempo mostra a variação de atributos como **dançabilidade**, **acústica** e **energia** por década.
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

# Filtragem de dados
filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]
filtered_df['decade'] = (filtered_df['year'] // 10 * 10).astype(int)

default_attributes = ["danceability", "acousticness", "energy"]

selected_attributes = st.multiselect(
    "Escolha os atributos para análise:",
    options=attributes,
    default=default_attributes 
)

decade_means = filtered_df.groupby('decade')[selected_attributes].mean().reset_index()

# Exibição do gráfico
st.write("### Evolução dos Atributos Acústicos")
plt.figure(figsize=(12, 6))
for attribute in selected_attributes:
    sns.lineplot(data=decade_means, x="decade", y=attribute, marker="o", label=attribute.capitalize())

plt.title(f"Evolução dos Atributos Acústicos por Década ({min_year}-{max_year})")
plt.xlabel("Década")
plt.ylabel("Média dos Atributos")
plt.legend(title="Atributos")

st.pyplot(plt)
