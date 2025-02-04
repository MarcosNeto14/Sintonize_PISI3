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

# Função para contar palavras-chave em um texto
def count_keywords(text, keywords_list):
    count = 0
    for word in keywords_list:
        count += text.lower().count(word.lower())
    return count

# Pré-processamento
df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
df['decade'] = (df['year'] // 10 * 10).astype('Int64')

attributes = [ 'violence', 'world/life', 'night/time', 'shake the audience', 'family/gospel', 'romantic', 'communication', 'obscene', 'music',
'movement/places', 'light/visual perceptions', 'family/spiritual', 'like/girls', 'sadness', 'feelings', 'danceability', 'loudness', 'acousticness',
'instrumentalness', 'valence', 'energy']
df[attributes] = df[attributes].apply(pd.to_numeric, errors='coerce')

# Menu lateral 
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "Visão Geral"

# Função para setar o menu
def set_menu(menu):
    st.session_state.current_menu = menu

st.sidebar.title("🎵 Sintonize")
st.sidebar.subheader("Selecione o detalhamento desejado abaixo:")

# Botões de navegação
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

# Recupera o menu atual
menu = st.session_state.current_menu

st.write(f"### {menu}")


if menu == "Visão Geral":
    st.title("📊 Análise da Evolução Musical ao Longo das Décadas")
    st.subheader("Dataset: Music Dataset : 1950 to 2019")
    st.subheader("Usabilidade: 9.41")
    st.markdown(""" O Sintonize utiliza um dataset contendo informações sobre músicas de diversas décadas, incluindo atributos como dançabilidade, energia, valência emocional e tristeza. A análise busca identificar padrões e tendências históricas, considerando eventos culturais e tecnológicos, como o impacto da internet nos gêneros musicais. """)
    
    col1, col2, col3 = st.columns(3)
    df['year'] = df['year'].astype('Int64')
    col1.metric("🎵 Total de Músicas", f"{len(df):,}".replace(",", "."))
    col2.metric("🗓️ Período Coberto", f"{int(df['year'].min())} a {int(df['year'].max())}")
    col3.metric("🎼 Gêneros Únicos", f"{df['genre'].nunique()}")
    st.write("### Amostra dos Dados")
    st.markdown(""" Esta tabela apresenta as **10 primeiras músicas** do dataset para uma visão inicial. Cada linha representa uma música e inclui informações como: - 🎤 **Artista:** Quem interpretou a música. - 🎵 **Nome da música:** O título da faixa. - 🗓️ **Ano de lançamento:** Quando a música foi lançada. - 📊 **Atributos musicais:** Dados como `danceability`, `energy`, e outros, que ajudam a descrever características sonoras. """)
    st.dataframe(df.head(10))
    st.write("#### Estatísticas Descritivas do Dataset")
    st.markdown(""" Esta tabela fornece uma visão geral das características numéricas do dataset. Aqui estão os detalhes: - **`count`**: Número de registros não nulos. - **`mean`**: Média dos valores, útil para entender tendências gerais. - **`std`**: Desvio padrão, indicando a variabilidade dos dados. - **`min` e `max`**: Os valores extremos registrados. - **`25%`, `50%`, `75%`**: Quartis, mostrando como os valores estão distribuídos. """)
    st.dataframe(df.describe())
elif menu == "Distribuição por Décadas": 
    st.markdown(""" Os gráficos mostram a quantidade de registros por gênero em cada década, ajudando a identificar as tendências musicais. - **Heatmap**: Facilita a comparação visual dos gêneros mais presentes ao longo do tempo. """)
    
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
    for decade, (start_year, end_year) in decades.items():
        filtered_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        if not filtered_df.empty:
            count_by_genre = filtered_df['genre'].value_counts()
            decade_data[decade] = count_by_genre
        else:
            decade_data[decade] = pd.Series(dtype=int)
    
    combined_data = pd.DataFrame(decade_data).fillna(0).astype(int)
    
    st.title("Distribuição por Décadas")
    
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

    # Atualiza os valores no session_state
    st.session_state.min_year = min_year
    st.session_state.max_year = max_year

    # Filtro de gêneros musicais
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = sorted(df['genre'].dropna().unique())  # Define todos como padrão

    selected_genres = st.multiselect(
        "Selecione os gêneros musicais para análise:",
        options=sorted(df['genre'].dropna().unique()),
        default=st.session_state.selected_genres  # Usa o filtro armazenado no session_state
    )

    # Atualiza os filtros no session_state
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
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(genre_distribution, annot=False, cmap="coolwarm", ax=ax)
    plt.xticks(rotation=0, ha='center')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')  # 'ha' centraliza o texto
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)
    ax.set_title(f"Distribuição de Gêneros por Década ({min_year}-{max_year})")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)


elif menu == "Evolução Acústica":
    st.markdown("""
    A linha do tempo mostra a variação de atributos como **dançabilidade**, **acústica** e **energia** por década. 
    Isso ajuda a observar como as características musicais evoluíram ao longo dos anos.
    """)

    df_filtered = df.dropna(subset=['year'] + attributes)

    min_year, max_year = st.slider(
        "Selecione o intervalo de anos para análise:",
        int(df['year'].min()), int(df['year'].max()), 
        (int(df['year'].min()), int(df['year'].max()))
    )

    filtered_df = df[(df['year'] >= min_year) & (df['year'] <= max_year)].copy()

    filtered_df['year'] = filtered_df['year'].astype(int)

    year_range = max_year - min_year

    if year_range >= 40:
        step = 10
    elif year_range >= 25:
        step = 5
    elif year_range >= 10:
        step = 2
    else:
        step = 1

    filtered_df['period'] = (filtered_df['year'] // step * step).astype(int)

    default_attributes = ["danceability", "acousticness", "energy"]

    selected_attributes = st.multiselect(
        "Escolha os atributos para análise:",
        options=attributes,
        default=default_attributes 
    )

    # Agrupando por período e calculando a média
    period_means = filtered_df.groupby('period')[selected_attributes].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for attribute in selected_attributes:
        sns.lineplot(data=period_means, x="period", y=attribute, marker="o", label=attribute.capitalize())

    plt.title(f"Evolução dos Atributos Acústicos ({min_year}-{max_year})")
    plt.xlabel("Ano")
    plt.ylabel("Média dos Atributos")
    plt.xticks(sorted(period_means['period'].unique()))  # Garante valores inteiros no eixo X
    plt.legend(title="Atributos")

    st.pyplot(plt)



if menu == "Palavras-chave e Contexto Histórico":
    # --- MOVIMENTOS SOCIAIS E DIREITOS CIVIS (1960-1970) ---
    st.markdown("### Movimentos Sociais e Direitos Civis (1960-1970)")
    st.markdown("Gráfico mostrando a frequência de palavras-chave relacionadas aos direitos civis e movimentos sociais nas letras das músicas dessa época, destacando o impacto cultural e social.")

    min_year_civil, max_year_civil = st.slider(
        " ",
        1950, 2019, (1960, 1970),
        key="slider_civil_rights"
    )

    filtered_df_civil_rights = df[(df['year'] >= min_year_civil) & (df['year'] <= max_year_civil)]
    keyword_counts_civil_rights = {year: 0 for year in range(min_year_civil, max_year_civil + 1)}

    keywords_civil_rights = [
        "freedom", "equality", "racism", "protest", "civil rights", "justice", "segregation", "peace", "discrimination", 
        "march", "activism", "black power", "revolution", "oppression", "human rights", "resistance", "liberation", 
        "empowerment", "suffrage", "inequality"
    ]

    for _, row in filtered_df_civil_rights.iterrows():
        lyrics = str(row['lyrics'])
        year = int(row['year'])
        keyword_counts_civil_rights[year] += count_keywords(lyrics, keywords_civil_rights)

    keyword_df_civil_rights = pd.DataFrame(list(keyword_counts_civil_rights.items()), columns=["Year", "Count"])
    keyword_df_civil_rights["Year"] = keyword_df_civil_rights["Year"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(keyword_df_civil_rights["Year"], keyword_df_civil_rights["Count"], color='lightgreen')
    ax.set_title(f"Frequência de Palavras-chave sobre Movimentos Sociais e Direitos Civis ({min_year_civil}-{max_year_civil})", fontsize=16)
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)

    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    st.pyplot(fig)


    # --- CHEGADA A LUA ---
    st.markdown("### Chegada à Lua (1967-1972)")
    st.markdown("Gráfico mostrando a frequência de palavras-chave relacionadas à missão Apollo nas letras, destacando o impacto cultural do evento.")

    min_year_moon, max_year_moon = st.slider(
        " ",
        1950, 2019, (1967, 1972),
        key="slider_moon"
    )

    filtered_df_moon = df[(df['year'] >= min_year_moon) & (df['year'] <= max_year_moon)]
    keyword_counts_moon = {year: 0 for year in range(min_year_moon, max_year_moon + 1)}

    for _, row in filtered_df_moon.iterrows():
        lyrics = str(row['lyrics'])
        year = row['year']
        keyword_counts_moon[year] += count_keywords(lyrics, keywords["moon_landing"])

    keyword_df_moon = pd.DataFrame(keyword_counts_moon.items(), columns=["Year", "Count"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(keyword_df_moon["Year"], keyword_df_moon["Count"], color='skyblue')
    ax.set_title(f"Frequência de Palavras-chave sobre a Chegada à Lua ({min_year_moon}-{max_year_moon})", fontsize=16)
    ax.set_xlabel("Ano", fontsize=12)
    ax.set_ylabel("Contagem de Palavras-chave", fontsize=12)
    st.pyplot(fig)

    # --- GUERRA FRIA ---
    st.markdown("### Guerra Fria (1950-1980)")
    st.markdown("Gráfico destacando palavras-chave sobre a Guerra Fria, mostrando sua influência em diferentes décadas.")

    min_year_cold, max_year_cold = st.slider(
        " ",
        1950, 2019, (1950, 1980),
        key="slider_cold_war"
    )

    filtered_df_cold_war = df[(df['year'] >= min_year_cold) & (df['year'] <= max_year_cold)]
    keyword_counts_cold_war = {year: 0 for year in range(min_year_cold, max_year_cold + 1)}

    for _, row in filtered_df_cold_war.iterrows():
        lyrics = str(row['lyrics'])
        year = row['year']
        keyword_counts_cold_war[year] += count_keywords(lyrics, keywords["cold_war"])

    keyword_df_cold_war = pd.DataFrame(keyword_counts_cold_war.items(), columns=["Year", "Count"])
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

    # Clusterização inicial (exemplo já existente no código)
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

    # Nova análise: Clusterização com 'sadness' e 'energy'
    st.markdown("""
    ### Atributos "Sadness" e "Energy"
    Outro agrupamento baseado em tristeza e energia, destacando faixas melancólicas ou intensas.
    """)

    acoustic_features = ['sadness', 'acousticness', 'energy']  # Alterando 'danceability' para 'sadness'
    acoustic_features = ['sadness', 'acousticness', 'energy']
    df_filtered_sadness = df.dropna(subset=acoustic_features)

    # Normalizar os dados para o modelo de clusterização
    scaler = StandardScaler()
    scaled_data_sadness = scaler.fit_transform(df_filtered_sadness[acoustic_features])

    # Realizar clusterização com K-Means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_filtered_sadness['cluster'] = kmeans.fit_predict(scaled_data_sadness)

    # Visualizar os clusters
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_filtered_sadness['sadness'], y=df_filtered_sadness['energy'], hue=df_filtered_sadness['cluster'], palette="Set1")
    plt.title("Clusterização de Músicas com Base em Atributos Acústicos", fontsize=16)
    plt.xlabel("Sadness", fontsize=12)
    plt.ylabel("Energy", fontsize=12)
    st.pyplot(plt)

elif menu == "Classificação de Gêneros":
    # Definir palavras-chave para cada gênero musical
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

    # Preparar os dados para a classificação
    X = []
    y = []

    # Para cada música, contar as palavras-chave de cada gênero
    for _, row in df.iterrows():
        lyrics = str(row['lyrics'])
        genre = row['genre']

        # Contar as palavras-chave de cada gênero
        genre_counts = {}
        for genre_name, genre_keywords in keywords.items():
            genre_counts[genre_name] = count_keywords(lyrics, genre_keywords)
        
        # Adicionar os dados ao conjunto X e o rótulo ao conjunto y
        X.append(list(genre_counts.values()))
        y.append(genre)

    # Converter X e y em DataFrame/arrays
    X = pd.DataFrame(X, columns=keywords.keys())
    y = pd.Series(y)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = rf.predict(X_test)

    # Exibir as métricas de avaliação
    report = classification_report(y_test, y_pred, target_names=keywords.keys(), output_dict=True)
    st.write("Métricas de Avaliação:")
    st.write(f"Precision, Recall, F1-Score e Support para cada Gênero:")
    st.write("Random Forest")

    # Mostrar as métricas de avaliação
    metrics_df = pd.DataFrame(report).transpose()
    st.dataframe(metrics_df)
    st.markdown("""
    A accuracy de 0.25 e a baixa precision, recall e F1-score tanto na macro quanto na weighted average indicam que o modelo está tendo 
    dificuldades para identificar corretamente os gêneros. 
    Ajustes futuros podem melhorar o desempenho ao adicionar mais palavras-chave ou ao treinar com dados mais equilibrados.
    """)