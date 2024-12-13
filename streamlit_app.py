import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import streamlit as st
import re

st.title("Análise de Impacto da Internet nos Gêneros Musicais")

parquet_path = r"C:\Users\Igor\Downloads\archive\tcc_ceds_music.parquet"

st.write("Carregando dados...")
try:
    df = pd.read_parquet(parquet_path)
except Exception as e:
    st.error(f"Erro ao carregar o arquivo Parquet: {e}")
    st.stop()

df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
df['decade'] = (df['year'] // 10 * 10).astype('Int64')

attributes = ['danceability', 'acousticness', 'energy']
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

decades = {
    "1950-1959": (1950, 1959),
    "1960-1969": (1960, 1969),
    "1970-1979": (1970, 1979),
    "1980-1989": (1980, 1989),
    "1990-1999": (1990, 1999),
    "2000-2009": (2000, 2009),
    "2010-2019": (2010, 2019),
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

st.write("### Comparação Detalhada Entre Décadas")
combined_data = pd.DataFrame(decade_data).fillna(0).astype(int)

colors = sns.color_palette("coolwarm", n_colors=len(combined_data.columns))[::-1]

fig, ax = plt.subplots(figsize=(12, 6))
combined_data.plot(kind='bar', ax=ax, color=colors)
ax.set_title("Distribuição de Gêneros por Década", fontsize=16)
ax.set_xlabel("Gêneros", fontsize=12)
ax.set_ylabel("Quantidade de Registros", fontsize=12)
ax.legend(title="Década", loc='upper left', bbox_to_anchor=(1.05, 1))
ax.set_xticklabels(combined_data.index, rotation=0)
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(combined_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
ax.set_title("Distribuição de Gêneros por Década (Heatmap)", fontsize=16)
ax.set_xlabel("Décadas", fontsize=12)
ax.set_ylabel("Gêneros", fontsize=12)
st.pyplot(fig)

st.write("### Evolução dos Atributos Acústicos por Década")

df_filtered = df.dropna(subset=['decade'] + attributes)

decade_means = df_filtered.groupby('decade')[attributes].mean().reset_index()

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

plt.title("Evolução dos Atributos Acústicos por Década", fontsize=16)
plt.xlabel("Década", fontsize=12)
plt.ylabel("Média dos Atributos", fontsize=12)
plt.legend(title="Atributos")
plt.xticks(decade_means["decade"], rotation=45)
plt.tight_layout()

st.pyplot(plt)

# Definir palavras-chave para diferentes acontecimentos
keywords = {
    "moon_landing": ["moon", "space", "NASA", "Apollo", "landing", "rocket", "moonwalk", "Neil Armstrong", "Buzz Aldrin"],
    "cold_war": ["cold war", "soviet", "missile", "freedom", "Khrushchev", "KGB", "sputnik", "nuclear", "Berlin", "communism"]
}

# Função para contar a ocorrência de palavras-chave em uma letra
def count_keywords(text, keywords_list):
    count = 0
    for word in keywords_list:
        count += text.lower().count(word.lower())
    return count

# Contar palavras-chave por ano (para o evento da lua entre 1967 e 1972)
keyword_counts_moon = {year: 0 for year in range(1967, 1973)}
for year in range(1967, 1973):
    filtered_df = df[(df['year'] == year)]
    for index, row in filtered_df.iterrows():
        lyrics = str(row['lyrics'])
        keyword_counts_moon[year] += count_keywords(lyrics, keywords["moon_landing"])

# Criar dataframe para visualização da chegada à Lua
keyword_df_moon = pd.DataFrame(keyword_counts_moon.items(), columns=["Year", "Count"])

# Visualizar com gráfico para a chegada à Lua
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(keyword_df_moon["Year"], keyword_df_moon["Count"], color='skyblue')
ax.set_title("Frequência de Palavras-chave sobre a Chegada à Lua (1967-1972)", fontsize=16)
ax.set_xlabel("Ano", fontsize=12)
ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)
st.pyplot(fig)

# Ajuste para mostrar as décadas no eixo X e não valores intermediários
keyword_counts_cold_war = {decade: 0 for decade in range(1950, 1990, 10)}  # Para décadas entre 1940-1980
for decade in range(1950, 1990, 10):
    start_year = decade
    end_year = decade + 9
    filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    for index, row in filtered_df.iterrows():
        lyrics = str(row['lyrics'])
        keyword_counts_cold_war[decade] += count_keywords(lyrics, keywords["cold_war"])

# Criar dataframe para visualização da Guerra Fria
keyword_df_cold_war = pd.DataFrame(keyword_counts_cold_war.items(), columns=["Decade", "Count"])

# Visualizar com gráfico para a Guerra Fria ajustado
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(keyword_df_cold_war["Decade"], keyword_df_cold_war["Count"], color='salmon')

# Ajustando os rótulos do eixo X para as décadas sem valores intermediários
ax.set_xticks(keyword_df_cold_war["Decade"])  # Exibir apenas os valores de década
ax.set_title("Frequência de Palavras-chave sobre a Guerra Fria", fontsize=16)
ax.set_xlabel("Década", fontsize=12)
ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)
st.pyplot(fig)

# Selecionar as colunas para clusterização
acoustic_features = ['danceability', 'acousticness', 'energy']
df_filtered = df.dropna(subset=acoustic_features)

# Normalizar os dados para o modelo de clusterização
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

# Realizar clusterização com K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

# Visualizar os clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
plt.title("Clusterização de Músicas com Base em Atributos Acústicos", fontsize=16)
st.markdown("""
### O que são os índices de "Danceability" e "Energy"?

- **Danceability** (dançabilidade): Mede a facilidade com que uma música pode ser dançada. O valor vai de **0.0 a 1.0**, onde **0.0** significa uma música difícil de dançar (como uma balada lenta) e **1.0** indica uma música muito dançante (como uma música de festa animada).
- **Energy** (energia): Reflete a intensidade da música. Um valor de **0.0** indica uma música calma e suave, enquanto **1.0** significa uma música energética e intensa, como uma música eletrônica de batida forte.

Divisão:

Músicas mais dançantes e energéticas: As músicas que possuem altos valores de danceability e energy estarão localizadas em áreas específicas do gráfico, geralmente com cores mais vibrantes, representando um cluster de músicas mais animadas e voltadas para festas.

Músicas suaves e menos energéticas: Músicas com valores baixos em ambos os atributos (danceability e energy) aparecem em outro grupo, possivelmente mais calmas ou introspectivas.

Músicas que não se enquadram especificamente nas situações anteriores.
""")

plt.xlabel("Danceability", fontsize=12)
plt.ylabel("Energy", fontsize=12)
st.pyplot(plt)

# Selecionar as colunas para clusterização
acoustic_features = ['sadness', 'acousticness', 'energy']  # Alterando 'danceability' para 'sadness'
df_filtered = df.dropna(subset=acoustic_features)

# Normalizar os dados para o modelo de clusterização
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

# Realizar clusterização com K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

# Visualizar os clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df_filtered['sadness'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
plt.title("Clusterização de Músicas com Base em Atributos Acústicos", fontsize=16)

st.markdown("""
### O que são os índices de "Sadness" e "Energy"?

- **Sadness** (tristeza): Mede o quão melancólica ou emocionalmente pesada uma música é. O valor vai de **0.0 a 1.0**, onde **0.0** indica uma música alegre e otimista, enquanto **1.0** indica uma música profunda e melancólica, com uma sensação de tristeza ou introspecção.
- **Energy** (energia): Reflete a intensidade da música. Um valor de **0.0** indica uma música calma e suave, enquanto **1.0** significa uma música energética e intensa, como uma música eletrônica de batida forte.

Divisão:

Músicas com alta tristeza e alta energia: Músicas que têm altos valores de tristeza e energia estarão localizadas em áreas específicas do gráfico, geralmente com cores mais vibrantes, representando músicas intensas e emocionais.

Músicas calmas e introspectivas: Músicas com valores baixos em ambos os atributos (sadness e energy) aparecem em outro grupo, possivelmente mais tranquilas ou melancólicas.

Músicas que não se enquadram especificamente nas situações anteriores.
""")

plt.xlabel("Sadness", fontsize=12)
plt.ylabel("Energy", fontsize=12)
st.pyplot(plt)


# Definir palavras-chave para cada gênero
keywords = {
    "blues": ["sad", "blues", "heartbreak", "crying", "rain"],
    "country": ["cowboy", "boots", "country", "ranch", "honky", "tumbleweed"],
    "pop": ["party", "love", "dance", "night", "club", "girl", "boy"],
    "hip_hop": ["rap", "street", "flow", "beat", "b-boy", "gangsta"],
    "jazz": ["improvisation", "saxophone", "swing", "blues", "jazz"],
    "reggae": ["rasta", "dub", "jah", "roots", "bob", "rastafari"],
    "rock": ["guitar", "rock", "band", "concert", "stage", "electric"]
}

# Função para contar a ocorrência de palavras-chave no texto da letra
def count_keywords(text, keywords_list):
    count = 0
    text = text.lower()
    for word in keywords_list:
        count += len(re.findall(r'\b' + re.escape(word) + r'\b', text))  # Contar palavras exatas
    return count

# Selecionar as colunas relevantes (letras e gêneros)
df_filtered = df.dropna(subset=['lyrics', 'genre'])

# Adicionar uma coluna com a contagem de palavras-chave para cada gênero
for genre, words in keywords.items():
    df_filtered[genre + '_count'] = df_filtered['lyrics'].apply(lambda x: count_keywords(x, words))

# Selecionar as features e o target
X = df_filtered[[genre + '_count' for genre in keywords.keys()]]
y = df_filtered['genre']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Fazer previsões
y_pred = clf.predict(X_test)

# Avaliar a performance do modelo
st.write("### Classificação de Gêneros Musicais com Base nas Letras")
st.markdown("""
A partir das letras das músicas, o modelo foi treinado para prever o gênero musical de uma música, utilizando palavras-chave específicas de cada gênero. As palavras-chave para cada gênero foram definidas com base em termos frequentemente associados ao estilo musical, como:

- **Blues**: palavras como "sad", "blues", "heartbreak".
- **Country**: palavras como "cowboy", "boots", "country".
- **Pop**: palavras como "party", "love", "dance".
- **Hip Hop**: palavras como "rap", "street", "flow".
- **Jazz**: palavras como "improvisation", "saxophone", "swing".
- **Reggae**: palavras como "rasta", "dub", "jah".
- **Rock**: palavras como "guitar", "rock", "band".

O modelo foi treinado para prever o gênero de uma música com base na presença dessas palavras-chave nas suas letras.

### Métricas de Performance
O modelo foi avaliado usando o **classification report**, que exibe as seguintes métricas:

- **Precision**: A precisão da classificação. Refere-se à quantidade de músicas classificadas corretamente como pertencentes a um gênero específico.
- **Recall**: A taxa de recuperação. Refere-se à porcentagem de músicas de um gênero que foram corretamente identificadas.
- **F1-score**: A média harmônica entre precisão e recall. Essa métrica ajuda a balancear a performance do modelo, especialmente quando temos um desequilíbrio entre os gêneros.
- **Support**: A quantidade de amostras de cada gênero no conjunto de testes.

Essas métricas são úteis para entender como o modelo está classificando os gêneros musicais com base nas letras e ajustar o modelo conforme necessário.

""")

# Exibir o relatório de classificação
st.write(classification_report(y_test, y_pred))

st.markdown("""
A accuracy de 0.25 e a baixa precision, recall e F1-score tanto na macro quanto na weighted average indicam que o modelo está tendo dificuldades para identificar corretamente os gêneros.
""")