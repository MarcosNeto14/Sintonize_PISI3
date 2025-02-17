import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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

# Página de Classificação de Gêneros
st.title("🎼 Classificação de Gêneros")
st.markdown("""
A análise a seguir utiliza **Random Forest** para classificar gêneros musicais com base em palavras-chave extraídas das letras.
""")

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

for _, row in df.iterrows():
    lyrics = str(row['lyrics'])
    genre = row['genre']
    genre_counts = {genre_name: count_keywords(lyrics, genre_keywords) for genre_name, genre_keywords in keywords.items()}
    X.append(list(genre_counts.values()))
    y.append(genre)

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
