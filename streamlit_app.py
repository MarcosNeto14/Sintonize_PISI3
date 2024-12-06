import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Análise de Impacto da Internet nos Gêneros Musicais")

csv_path = r"C:\Users\Marcos\Downloads\archive\tcc_ceds_music.csv"

# Carregar dataset com verificação do delimitador
try:
    with open(csv_path, 'r') as file:
        first_line = file.readline()
    delimiter = ',' if ',' in first_line else ';'

    df = pd.read_csv(csv_path, delimiter=delimiter)
except Exception as e:
    st.stop()

# Preprocessar os dados
df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
df['decade'] = (df['year'] // 10 * 10).astype('Int64')

# Garantir que as colunas de atributos acústicos sejam numéricas
attributes = ['danceability', 'acousticness', 'energy']
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

# Análise de gêneros por década (mantendo sua lógica)
decades = {
    "1960-1969": (1960, 1969),
    "1970-1979": (1970, 1979),
    "1980-1989": (1980, 1989),
    "1990-1999": (1990, 1999),
    "2000-2009": (2000, 2009),
}

decade_data = {}
for decade, (start_year, end_year) in decades.items():
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    st.write(f"#### {decade}: {len(filtered_df)} registros encontrados")
    if not filtered_df.empty:
        count_by_genre = filtered_df['genre'].value_counts()
        decade_data[decade] = count_by_genre
    else:
        decade_data[decade] = pd.Series(dtype=int)

st.write("### Comparação de Gêneros por Década")
for decade, counts in decade_data.items():
    st.write(f"#### {decade}")
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind='bar', ax=ax)
    ax.set_title(f"Gêneros em {decade}")
    ax.set_ylabel("Quantidade")
    ax.set_xticklabels(counts.index, rotation=0)  
    st.pyplot(fig)

st.write("### Comparação Detalhada Entre Décadas")
combined_data = pd.DataFrame(decade_data).fillna(0).astype(int)
st.write(combined_data)

fig, ax = plt.subplots(figsize=(12, 6))
combined_data.plot(kind='bar', ax=ax)
ax.set_title("Distribuição de Gêneros por Década", fontsize=16)
ax.set_xlabel("Gêneros", fontsize=12)
ax.set_ylabel("Quantidade de Registros", fontsize=12)
ax.legend(title="Década")
ax.set_xticklabels(combined_data.index, rotation=0)  
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(combined_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
ax.set_title("Distribuição de Gêneros por Década (Heatmap)", fontsize=16)
ax.set_xlabel("Décadas", fontsize=12)
ax.set_ylabel("Gêneros", fontsize=12)
st.pyplot(fig)

# NOVA SEÇÃO: Evolução dos atributos acústicos por década
st.write("### Evolução dos Atributos Acústicos por Década")

# Garantir que os dados acústicos não sejam nulos ou não numéricos
df_filtered = df.dropna(subset=['decade'] + attributes)

# Calcular a média por década
decade_means = df_filtered.groupby('decade')[attributes].mean().reset_index()

# Plotar gráficos lineares
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
for attribute in attributes:
    sns.lineplot(
        data=decade_means,
        x="decade",
        y=attribute,
        marker="o",
        label=attribute.capitalize()
    )

# Personalizar o gráfico
plt.title("Evolução dos Atributos Acústicos por Década", fontsize=16)
plt.xlabel("Década", fontsize=12)
plt.ylabel("Média dos Atributos", fontsize=12)
plt.legend(title="Atributos")
plt.xticks(decade_means["decade"], rotation=45)
plt.tight_layout()

# Mostrar no Streamlit
st.pyplot(plt)
