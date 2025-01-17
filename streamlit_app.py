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

st.set_page_config(page_title="Sintonize", layout="wide")

base_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(base_dir, "dataset/parquet/tcc_ceds_music.parquet")

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

keywords = {
    "moon_landing": ["moon", "space", "NASA", "Apollo", "landing", "rocket", "moonwalk", "Neil Armstrong", "Buzz Aldrin"],
    "cold_war": ["cold war", "soviet", "missile", "freedom", "Khrushchev", "KGB", "sputnik", "nuclear", "Berlin", "communism"]
}

def count_keywords(text, keywords_list):
    count = 0
    for word in keywords_list:
        count += text.lower().count(word.lower())
    return count

df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
df['decade'] = (df['year'] // 10 * 10).astype('Int64')

attributes = ['danceability', 'acousticness', 'energy']
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

if "current_menu" not in st.session_state:
    st.session_state.current_menu = "Visão Geral"

def set_menu(menu):
    st.session_state.current_menu = menu

st.sidebar.title("🎵 Sintonize")
st.sidebar.subheader("Selecione o detalhamento desejado abaixo:")
if st.sidebar.button("📄 Visão Geral"):
    set_menu("Visão Geral")
if st.sidebar.button("📊  Distribuição por Décadas"):
    set_menu("Distribuição por Décadas")
if st.sidebar.button("🎶  Evolução Acústica"):
    set_menu("Evolução Acústica")
if st.sidebar.button("🌌 Palavras-chave e Contexto Histórico"):
    set_menu("Palavras-chave e Contexto Histórico")
if st.sidebar.button("🔗  Clusterização"):
    set_menu("Clusterização")
if st.sidebar.button("🎼  Classificação de Gêneros"):
    set_menu("Classificação de Gêneros")

menu = st.session_state.current_menu

st.write(f"### {menu}")

if menu == "Visão Geral":
    st.title("📊 Análise da Evolução Musical ao Longo das Décadas")
    st.subheader("Dataset: Music Dataset : 1950 to 2019")
    st.subheader("Usabilidade: 9.41")
    st.markdown("""
    O Sintonize utiliza um dataset contendo informações sobre músicas de diversas décadas, incluindo atributos como dançabilidade, energia, valência emocional e tristeza. 
    A análise busca identificar padrões e tendências históricas, considerando eventos culturais e tecnológicos, como o impacto da internet nos gêneros musicais.
    """)

    col1, col2, col3 = st.columns(3)
    df['year'] = df['year'].astype('Int64')
    col1.metric("🎵 Total de Músicas", f"{len(df):,}".replace(",", "."))
    col2.metric("🗓️ Período Coberto", f"{int(df['year'].min())} a {int(df['year'].max())}")
    col3.metric("🎼 Gêneros Únicos", f"{df['genre'].nunique()}")
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
    st.dataframe(df.describe())
elif menu == "Distribuição por Décadas":
    st.markdown("""
    Os gráficos mostram a quantidade de registros por gênero em cada década, ajudando a identificar as tendências musicais.
    """)

    st.title("Distribuição por Décadas")
    
    # Seleção do intervalo de anos
    min_year, max_year = st.slider(
        "Selecione o intervalo de anos para análise:",
        int(df['year'].min()), int(df['year'].max()),
        (int(df['year'].min()), int(df['year'].max()))
    )
    
    # Filtro por gênero
    available_genres = sorted(df['genre'].dropna().unique())
    selected_genres = st.multiselect(
        "Selecione os gêneros para análise:",
        options=available_genres,
        default=available_genres  # Por padrão, inclui todos os gêneros
    )
    
    # Aplicar filtros ao dataframe
    filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]
    if selected_genres:
        filtered_df = filtered_df[filtered_df['genre'].isin(selected_genres)]
    
    # Agrupamento por década e gênero
    filtered_df['decade'] = (filtered_df['year'] // 10 * 10).astype(int)
    genre_distribution = filtered_df.groupby(['decade', 'genre']).size().unstack(fill_value=0)
    
    # Criação do gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_distribution, cmap="coolwarm", annot=False, fmt="d", ax=ax)
    ax.set_title(f"Distribuição de Gêneros por Década ({min_year}-{max_year})")
    st.pyplot(fig)

elif menu == "Evolução Acústica":
    st.markdown("""
    A linha do tempo mostra a variação de atributos como **dançabilidade**, **acústica** e **energia** por década. 
Isso ajuda a observar como as características musicais evoluíram ao longo dos anos.
    """)
    min_year, max_year = st.slider(
        "Selecione o intervalo de anos para análise:",
        int(df['year'].min()), int(df['year'].max()), 
        (int(df['year'].min()), int(df['year'].max()))
    )
    filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)]
    filtered_df['decade'] = (filtered_df['year'] // 10 * 10).astype(int)
    
    decade_means = filtered_df.groupby('decade')[attributes].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    for attribute in attributes:
        sns.lineplot(data=decade_means, x="decade", y=attribute, marker="o", label=attribute.capitalize())
    plt.title(f"Evolução dos Atributos Acústicos por Década ({min_year}-{max_year})")
    plt.xlabel("Década")
    plt.ylabel("Média dos Atributos")
    plt.legend(title="Atributos")
    st.pyplot(plt)
elif menu == "Palavras-chave e Contexto Histórico":
    st.markdown("""
    ### Chegada à Lua
    Gráfico mostrando a frequência de palavras-chave relacionadas à missão Apollo nas letras, destacando o impacto cultural do evento.
    """)

    min_year_moon, max_year_moon = st.slider(
        "Selecione o intervalo de anos para análise (Chegada à Lua):",
        1950, 2019, (1950, 2019)
    )
    filtered_df_moon = df[(df['year'] >= min_year_moon) & (df['year'] <= max_year_moon)]
    keyword_counts_moon = {year: 0 for year in range(min_year_moon, max_year_moon + 1)}

    for year in range(min_year_moon, max_year_moon + 1):
        year_df = filtered_df_moon[filtered_df_moon['year'] == year]
        for _, row in year_df.iterrows():
            lyrics = str(row['lyrics'])
            keyword_counts_moon[year] += count_keywords(lyrics, keywords["moon_landing"])

    keyword_df_moon = pd.DataFrame(keyword_counts_moon.items(), columns=["Year", "Count"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(keyword_df_moon["Year"], keyword_df_moon["Count"], color='skyblue')
    ax.set_title(f"Frequência de Palavras-chave sobre a Chegada à Lua ({min_year_moon}-{max_year_moon})", fontsize=16)
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)
    st.pyplot(fig)

    st.markdown("""
    ### Guerra Fria (1950-1980)
    Gráfico destacando palavras-chave sobre a Guerra Fria, mostrando sua influência em diferentes anos.
    """)

    # Filtro para Guerra Fria
    min_year_cold, max_year_cold = st.slider(
        "Selecione o intervalo de anos para análise (Guerra Fria):",
        1950, 2019, (1950, 2019)
    )
    filtered_df_cold_war = df[(df['year'] >= min_year_cold) & (df['year'] <= max_year_cold)]
    keyword_counts_cold_war = {year: 0 for year in range(min_year_cold, max_year_cold + 1)}

    for year in range(min_year_cold, max_year_cold + 1):
        year_df = filtered_df_cold_war[filtered_df_cold_war['year'] == year]
        for _, row in year_df.iterrows():
            lyrics = str(row['lyrics'])
            keyword_counts_cold_war[year] += count_keywords(lyrics, keywords["cold_war"])

    keyword_df_cold_war = pd.DataFrame(keyword_counts_cold_war.items(), columns=["Year", "Count"])

    # Gráfico para Guerra Fria
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(keyword_df_cold_war["Year"], keyword_df_cold_war["Count"], color='salmon')
    ax.set_title(f"Frequência de Palavras-chave sobre a Guerra Fria ({min_year_cold}-{max_year_cold})", fontsize=16)
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)
    st.pyplot(fig)

elif menu == "Clusterização":
    st.markdown("""
    ### Agrupamento Baseado em Atributos Acústicos
    Utilizando K-Means para agrupar músicas com base em seus atributos acústicos, destacando padrões e semelhanças.
    """)

    acoustic_features = ['danceability', 'energy', 'acousticness']
    df_filtered = df.dropna(subset=acoustic_features)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_filtered[acoustic_features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered['cluster'] = kmeans.fit_predict(scaled_data)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_filtered['danceability'], y=df_filtered['energy'], hue=df_filtered['cluster'], palette="Set1")
    plt.title("Clusterização de Músicas com Base em Danceability e Energy", fontsize=16)
    plt.xlabel("Danceability", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    st.pyplot(plt)

    st.markdown("""
    ### Atributos "Sadness" e "Energy"
    Outro agrupamento baseado em tristeza e energia, destacando faixas melancólicas ou intensas.
    """)

    acoustic_features = ['sadness', 'acousticness', 'energy']
    df_filtered_sadness = df.dropna(subset=acoustic_features)

    scaler = StandardScaler()
    scaled_data_sadness = scaler.fit_transform(df_filtered_sadness[acoustic_features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered_sadness['cluster'] = kmeans.fit_predict(scaled_data_sadness)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_filtered_sadness['sadness'], y=df_filtered_sadness['energy'], hue=df_filtered_sadness['cluster'], palette="Set1")
    plt.title("Clusterização de Músicas com Base em Atributos Acústicos", fontsize=16)
    plt.xlabel("Sadness", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    st.pyplot(plt)

elif menu == "Classificação de Gêneros":
    keywords = {
        "blues": ["sad", "blues", "heartbreak", "crying", "rain"],
        "country": ["cowboy", "boots", "country", "ranch", "honky", "tumbleweed"],
        "pop": ["party", "love", "dance", "night", "club", "girl", "boy"],
        "hip_hop": ["rap", "street", "flow", "beat", "b-boy", "gangsta"],
        "jazz": ["improvisation", "saxophone", "swing", "blues", "jazz"],
        "reggae": ["rasta", "dub", "jah", "roots", "bob", "rastafari"],
        "rock": ["guitar", "rock", "band", "concert", "stage", "electric"]
    }

    
    df = df.dropna(subset=['lyrics', 'genre'])
    df['lyrics'] = df['lyrics'].fillna('')

    X = []
    y = []

    for _, row in df.iterrows():
        lyrics = str(row['lyrics'])
        genre = row['genre']

        genre_counts = {}
        for genre_name, genre_keywords in keywords.items():
            genre_counts[genre_name] = count_keywords(lyrics, genre_keywords)

        X.append(list(genre_counts.values()))
        y.append(genre)

    X = pd.DataFrame(X, columns=keywords.keys())
    y = pd.Series(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=keywords.keys(), output_dict=True)
    st.write("Métricas de Avaliação:")
    st.write(f"Precision, Recall, F1-Score e Support para cada Gênero:")
    st.write("Random Forest")

    metrics_df = pd.DataFrame(report).transpose()
    st.dataframe(metrics_df)
    st.markdown("""
    A accuracy de 0.25 e a baixa precision, recall e F1-score tanto na macro quanto na weighted average indicam que o modelo está tendo 
    dificuldades para identificar corretamente os gêneros. 
    Ajustes futuros podem melhorar o desempenho ao adicionar mais palavras-chave ou ao treinar com dados mais equilibrados.
    """)

