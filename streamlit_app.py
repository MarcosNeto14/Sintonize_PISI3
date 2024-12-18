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
import re

# Configurações iniciais
st.set_page_config(page_title="Sintonize", layout="wide")

# Caminho absoluto baseado no local do script
base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, "dataset/parquet/tcc_ceds_music.parquet")

# Carregando os dados
@st.cache_data  # Cache para evitar recarregamento
def load_data(path):
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

df = load_data(parquet_path)
if df is None:
    st.stop()

# Pré-processamento
df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
df['decade'] = (df['year'] // 10 * 10).astype('Int64')

attributes = ['danceability', 'acousticness', 'energy']
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

# Menu lateral com botões
st.sidebar.title("🎵 Sintonize")
st.sidebar.subheader("Selecione o detalhamento desejado abaixo:")
menu = None  # Variável para guardar o estado do menu

if st.sidebar.button("📄 Visão Geral"):
    menu = "Visão Geral"
if st.sidebar.button("📊  Distribuição por Décadas"):
    menu = "Distribuição por Décadas"
if st.sidebar.button("🎶  Evolução Acústica"):
    menu = "Evolução Acústica"
if st.sidebar.button("🔗  Clusterização"):
    menu = "Clusterização"
if st.sidebar.button("🎼  Classificação de Gêneros"):
    menu = "Classificação de Gêneros"

# Caso nenhum botão seja clicado, definir um padrão
if not menu:
    menu = "Visão Geral"  # Menu padrão inicial

# Mostrar a análise correspondente
st.write(f"###  {menu}")

# Separando as partes do menu
if menu == "Visão Geral":
    # Título e subtítulo
    st.title("📊 Análise do Impacto Histórico na Ascensão e Queda de Gêneros Musicais")
    st.subheader("Explorando tendências musicais ao longo das décadas")

    # Descrição
    st.markdown("""
    O Sintonize utiliza um dataset contendo informações sobre músicas de diversas décadas, incluindo atributos como dançabilidade, energia, valência emocional e tristeza. 
    A análise busca identificar padrões e tendências históricas, considerando eventos culturais e tecnológicos, como o impacto da internet nos gêneros musicais.
    """)

    # Informações básicas do dataset com visualização em coluna
    col1, col2, col3 = st.columns(3)
    # Corrigir o tipo da coluna 'year'
    df['year'] = df['year'].astype('Int64')
    # Agora sim criando as colunas
    col1.metric("🎵 Total de Músicas", f"{len(df):,}".replace(",", "."))
    col2.metric("🗓️ Período Coberto", f"{int(df['year'].min())} a {int(df['year'].max())}")
    col3.metric("🎼 Gêneros Únicos", f"{df['genre'].nunique()}")

    # Amostra dos dados
    st.write("### Amostra dos Dados")
    st.markdown("""
    Esta tabela apresenta as **10 primeiras músicas** do dataset para uma visão inicial. 
    Cada linha representa uma música e inclui informações como:
    - 🎤 **Artista:** Quem interpretou a música.
    - 🎵 **Nome da música:** O título da faixa.
    - 🗓️ **Ano de lançamento:** Quando a música foi lançada.
    - 📊 **Atributos musicais:** Dados como `danceability`, `energy`, e outros, que ajudam a descrever características sonoras.
    """)
    st.dataframe(df.head(10))

    #Comentei pois me pareceu duplicação
    #st.write("### Visão Geral dos Dados")
    #st.write(df.head(10))
    # Comentei e substituí por "Estatísticas Descritivas do Dataset"
    #st.write("**Informações do Dataset:**")
    #st.write(df.describe())

    # Estatísticas Descritivas do Dataset
    st.write("#### Estatísticas Descritivas do Dataset")
    st.markdown("""
    Esta tabela fornece uma visão geral das características numéricas do dataset. 
    Aqui estão os detalhes:
    - **`count`**: Número de registros não nulos.
    - **`mean`**: Média dos valores, útil para entender tendências gerais.
    - **`std`**: Desvio padrão, indicando a variabilidade dos dados.
    - **`min` e `max`**: Os valores extremos registrados.
    - **`25%`, `50%`, `75%`**: Quartis, mostrando como os valores estão distribuídos.
    """)

    # Exibir tabela descritiva
    st.dataframe(df.describe())

    # Gráficos de distribuição (Histograma) das principais métricas
    st.write("#### Distribuição dos Atributos")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))  # Criando 3 subplots lado a lado
    attributes = ['danceability', 'energy', 'acousticness']

    for i, attr in enumerate(attributes):
        ax[i].hist(df[attr].dropna(), bins=20, color="skyblue", edgecolor="black")
        ax[i].set_title(f"Distribuição de {attr.capitalize()}")
        ax[i].set_xlabel(attr.capitalize())
        ax[i].set_ylabel("Frequência")

    st.pyplot(fig)

# Distribuição por Décadas
elif menu == "Distribuição por Décadas":
    st.write("### Distribuição de Gêneros por Década")
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
    combined_data = pd.DataFrame(decade_data).fillna(0).astype(int)

    # Visualização
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(combined_data, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Distribuição de Gêneros por Década (Heatmap)")
    st.pyplot(fig)

# Evolução Acústica
elif menu == "Evolução Acústica":
    st.write("### Evolução dos Atributos Acústicos")
    df_filtered = df.dropna(subset=['decade'] + attributes)
    decade_means = df_filtered.groupby('decade')[attributes].mean().reset_index()

    # Gráfico
    plt.figure(figsize=(12, 6))
    for attribute in attributes:
        sns.lineplot(data=decade_means, x="decade", y=attribute, marker="o", label=attribute.capitalize())
    plt.title("Evolução dos Atributos Acústicos por Década")
    plt.xlabel("Década")
    plt.ylabel("Média dos Atributos")
    plt.legend(title="Atributos")
    st.pyplot(plt)

# Clusterização
elif menu == "Clusterização":
    st.write("### Clusterização de Músicas")
    acoustic_features = ['danceability', 'acousticness', 'energy']
    df_filtered = df.dropna(subset=acoustic_features)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
    plt.title("Clusterização de Músicas (Danceability vs Energy)")
    plt.xlabel("Danceability")
    plt.ylabel("Energy")
    st.pyplot(plt)

# Classificação de Gêneros
elif menu == "Classificação de Gêneros":
    st.write("### Classificação de Gêneros Musicais")
    # Seleção de palavras-chave e treino de modelo
    keywords = {
        "pop": ["party", "love", "dance", "night"],
        "rock": ["guitar", "rock", "band"],
    }
    def count_keywords(text, keywords_list):
        count = 0
        for word in keywords_list:
            count += len(re.findall(r'\b' + re.escape(word) + r'\b', text.lower()))
        return count

    df_filtered = df.dropna(subset=['lyrics', 'genre'])
    for genre, words in keywords.items():
        df_filtered[genre + '_count'] = df_filtered['lyrics'].apply(lambda x: count_keywords(x, words))

    X = df_filtered[[genre + '_count' for genre in keywords.keys()]]
    y = df_filtered['genre']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.write("### Relatório de Classificação")
    st.text(classification_report(y_test, y_pred))