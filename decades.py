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

# Definição das décadas
decades = {
    "1950-1959": (1950, 1959),
    "1960-1969": (1960, 1969),
    "1970-1979": (1970, 1979),
    "1980-1989": (1980, 1989),
    "1990-1999": (1990, 1999),
    "2000-2009": (2000, 2009),
    "2010-2019": (2010, 2019),
}

# Mapeamento por década
decade_data = {}
for decade, (start, end) in decades.items():
    filtered_df = df[(df['year'] >= start) & (df['year'] <= end)]
    count_by_genre = filtered_df['genre'].value_counts()
    decade_data[decade] = count_by_genre

combined_data = pd.DataFrame(decade_data).fillna(0).astype(int)

# Exibição do Heatmap
st.write("### Heatmap de Gêneros por Década")
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(combined_data, annot=False, cmap="coolwarm", ax=ax)
plt.xticks(rotation=0, ha='center')
ax.set_title("Distribuição de Gêneros por Década")
st.pyplot(fig)
