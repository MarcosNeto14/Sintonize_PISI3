import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, count_keywords

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

# Página de Palavras-chave e Contexto Histórico
st.title("🌌 Palavras-chave e Contexto Histórico")
st.markdown("""
A análise a seguir mostra a frequência de palavras-chave relacionadas a eventos históricos dentro das letras das músicas.
""")

# Definição de palavras-chave
keywords = {
    "moon_landing": ["moon", "space", "NASA", "Apollo", "landing", "rocket", "moonwalk", "Neil Armstrong", "Buzz Aldrin"],
    "cold_war": ["cold war", "soviet", "missile", "freedom", "Khrushchev", "KGB", "sputnik", "nuclear", "Berlin", "communism"]
}

# Seleção de evento histórico
event = st.selectbox("Selecione um evento histórico:", list(keywords.keys()))

df_filtered = df.dropna(subset=["lyrics", "year"])
keyword_counts = {year: 0 for year in df_filtered["year"].unique()}

for _, row in df_filtered.iterrows():
    lyrics = str(row["lyrics"])
    year = row["year"]
    keyword_counts[year] += count_keywords(lyrics, keywords[event])

# Criando DataFrame para visualização
keyword_df = pd.DataFrame(keyword_counts.items(), columns=["Year", "Count"]).sort_values("Year")

# Gráfico de frequência de palavras-chave
st.write(f"### Frequência de Palavras-chave: {event.replace('_', ' ').title()}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(keyword_df["Year"], keyword_df["Count"], color='skyblue')
ax.set_xlabel("Ano")
ax.set_ylabel("Contagem de Palavras-chave")
ax.set_title(f"Frequência de Palavras-chave sobre {event.replace('_', ' ').title()}")
st.pyplot(fig)
