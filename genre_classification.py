import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from utils import load_data, count_keywords

# Caminho do dataset
parquet_path = "dataset/parquet/tcc_ceds_music.parquet"
df = load_data(parquet_path)
if df is None:
    st.stop()

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
Esta aplicação permite classificar gêneros musicais com base em palavras-chave extraídas das letras. 
Escolha um modelo de classificação, insira palavras-chave ou uma letra de música e veja o gênero previsto.
""")

# Dicionário de palavras-chave por gênero
keywords = {
    "blues": ["sad", "blues", "heartbreak", "crying", "rain", "whiskey", "trouble", "lonely", "devil", "soul", "pain"],
    "country": ["cowboy", "boots", "country", "ranch", "honky", "tumbleweed", "truck", "whiskey", "barn", "horse", "southern", "fiddle"],
    "pop": ["party", "love", "dance", "night", "club", "girl", "boy", "baby", "radio", "top", "dream", "kiss", "music", "heart"],
    "hip_hop": ["rap", "street", "flow", "beat", "b-boy", "gangsta", "money", "hustle", "rhymes", "mic", "crew", "trap", "real"],
    "jazz": ["improvisation", "saxophone", "swing", "blues", "jazz", "piano", "trumpet", "bass", "chords", "melody", "harmony", "scat"],
    "reggae": ["rasta", "dub", "jah", "roots", "bob", "rastafari", "ganja", "irie", "vibes", "marley", "sunshine", "peace", "one love"],
    "rock": ["guitar", "rock", "band", "concert", "stage", "electric", "drums", "solo", "riff", "headbang", "loud", "rebel", "live"]
}

# Preparar os dados
df = df.dropna(subset=['lyrics', 'genre'])
df['lyrics'] = df['lyrics'].fillna('')

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

# =====================================
# Seção 1: Escolha do Modelo
# =====================================
st.write("### 🔧 Escolha do Modelo")
model_choice = st.selectbox(
    "Selecione o modelo de classificação:",
    ["Random Forest", "SVM", "SVM + SMOTE", "KNN + Undersampling"],
    help="Escolha um modelo para classificar os gêneros musicais."
)

# Configurar o modelo selecionado
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_choice == "SVM":
    model = SVC(kernel='linear', random_state=42)
elif model_choice == "SVM + SMOTE":
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    model = SVC(kernel='linear', random_state=42)
elif model_choice == "KNN + Undersampling":
    undersampler = RandomUnderSampler(random_state=42)
    X_train, y_train = undersampler.fit_resample(X_train, y_train)
    model = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo
model.fit(X_train, y_train)

# =====================================
# Seção 2: Simulação de Classificação
# =====================================
st.write("### 🎵 Simulação de Classificação")
st.markdown("""
Insira palavras-chave ou uma letra de música para classificar o gênero.
""")

# Input do usuário
user_input = st.text_area("Insira palavras-chave ou uma letra de música:")

if user_input:
    # Contar palavras-chave no input do usuário
    user_counts = {genre_name: count_keywords(user_input, genre_keywords) for genre_name, genre_keywords in keywords.items()}
    user_X = pd.DataFrame([list(user_counts.values())], columns=keywords.keys())

    # Prever o gênero
    predicted_genre = model.predict(user_X)
    st.success(f"### 🎶 O gênero previsto é: **{predicted_genre[0]}**")

# =====================================
# Seção 3: Avaliação do Modelo
# =====================================
st.write("### 📊 Avaliação do Modelo")
st.markdown("""
Abaixo estão as métricas de avaliação do modelo selecionado.
""")

# Avaliar o modelo
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=keywords.keys(), output_dict=True)
metrics_df = pd.DataFrame(report).transpose()

st.write("Métricas de Avaliação:")
st.dataframe(metrics_df)

st.markdown("""
- **Precision**: A proporção de previsões corretas para um gênero específico.
- **Recall**: A proporção de instâncias corretamente classificadas de um gênero.
- **F1-Score**: A média harmônica entre precision e recall.
- **Support**: O número de ocorrências de cada gênero no conjunto de teste.
""")