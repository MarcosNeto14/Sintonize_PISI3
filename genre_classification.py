import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
A análise a seguir permite selecionar diferentes modelos para classificar gêneros musicais com base em palavras-chave extraídas das letras.
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

# Criar opção de seleção de modelo
model_choice = st.selectbox("Escolha o modelo de classificação:", ["Random Forest", "SVM", "SVM + SMOTE", "KNN + Undersampling"])

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

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Exibir as métricas de avaliação
report = classification_report(y_test, y_pred, target_names=keywords.keys(), output_dict=True)
st.write("Métricas de Avaliação:")
st.write(f"Precision, Recall, F1-Score e Support para cada Gênero:")
st.write(f"{model_choice}")

# Mostrar as métricas de avaliação
metrics_df = pd.DataFrame(report).transpose()
st.dataframe(metrics_df)

st.markdown("""
A accuracy e as métricas indicam o desempenho do modelo selecionado na classificação dos gêneros musicais.
Ajustes futuros podem melhorar o desempenho ao adicionar mais palavras-chave ou ao treinar com dados mais equilibrados.
""")
