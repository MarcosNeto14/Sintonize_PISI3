import streamlit as st
import pandas as pd
import shap
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

# Usar o dataset completo, sem filtrar por gênero
df_filtered = df  # Remova o filtro por gênero

# Preparar os dados
df_filtered = df_filtered.dropna(subset=['lyrics', 'genre'])
df_filtered['lyrics'] = df_filtered['lyrics'].fillna('')

X = []
y = []

for _, row in df_filtered.iterrows():
    lyrics = str(row['lyrics'])
    genre = row['genre']
    genre_counts = {genre_name: count_keywords(lyrics, genre_keywords) for genre_name, genre_keywords in keywords.items()}
    X.append(list(genre_counts.values()))
    y.append(genre)

X = pd.DataFrame(X, columns=keywords.keys())
y = pd.Series(y)

# Visualizar a distribuição das classes antes do balanceamento
st.write("### 📊 Distribuição das Classes Antes do Balanceamento")
class_distribution_before = y.value_counts()
plt.figure(figsize=(10, 6))
class_distribution_before.plot(kind='barh', color='skyblue')  # Gráfico horizontal
plt.xlabel("Número de Exemplos")
plt.ylabel("Gênero Musical")
plt.title("Distribuição das Classes Antes do Balanceamento")
st.pyplot(plt)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Aplicar SMOTE para balancear as classes no conjunto de treinamento
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Visualizar a distribuição das classes após o balanceamento
st.write("### 📊 Distribuição das Classes Após o Balanceamento (SMOTE)")
class_distribution_after = pd.Series(y_train_resampled).value_counts()
plt.figure(figsize=(10, 6))
class_distribution_after.plot(kind='barh', color='lightgreen')  # Gráfico horizontal
plt.xlabel("Número de Exemplos")
plt.ylabel("Gênero Musical")
plt.title("Distribuição das Classes Após o Balanceamento")
st.pyplot(plt)

# =====================================
# Seção 2: Escolha do Modelo
# =====================================
st.write("### 🔧 Escolha do Modelo")
model_choice = st.selectbox(
    "Selecione o modelo de classificação:",
    ["Random Forest", "SVM", "SVM + SMOTE", "KNN + Undersampling"],
    help="Escolha um modelo para classificar os gêneros musicais.",
    key="model_choice_selectbox"  # Adicionando uma chave única
)

# Configurar o modelo selecionado
if model_choice == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
elif model_choice == "SVM":
    model = SVC(kernel='linear', random_state=42, probability=True)
elif model_choice == "SVM + SMOTE":
    model = SVC(kernel='linear', random_state=42, probability=True)
elif model_choice == "KNN + Undersampling":
    undersampler = RandomUnderSampler(random_state=42)
    X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)
    model = KNeighborsClassifier(n_neighbors=5)

# Treinar o modelo com os dados balanceados
model.fit(X_train_resampled, y_train_resampled)

# =====================================
# Seção 3: Avaliação do Modelo
# =====================================
st.write("### 📊 Avaliação do Modelo")
st.markdown("""
Abaixo estão as métricas de avaliação do modelo selecionado.
""")

# Avaliar o modelo
y_pred = model.predict(X_test)

# Verifique os gêneros presentes em y_test
present_genres = y_test.unique()

# Gere o relatório de classificação apenas com os gêneros presentes
report = classification_report(y_test, y_pred, target_names=present_genres, output_dict=True)

# Extrair as métricas para cada classe, média e média ponderada
metrics = {
    'Precisão': [],
    'Recall': [],
    'F1-Score': [],
    'Suporte': [],
    'Acurácia': []  # Adicionando Acurácia como uma métrica
}

# Preencher as métricas para cada classe presente
for genre in present_genres:
    metrics['Precisão'].append(report[genre]['precision'])
    metrics['Recall'].append(report[genre]['recall'])
    metrics['F1-Score'].append(report[genre]['f1-score'])
    metrics['Suporte'].append(report[genre]['support'])
    metrics['Acurácia'].append(None)  # Acurácia não é por classe, então deixamos como None

# Adicionar as métricas de média
metrics['Precisão'].append(report['macro avg']['precision'])
metrics['Recall'].append(report['macro avg']['recall'])
metrics['F1-Score'].append(report['macro avg']['f1-score'])
metrics['Suporte'].append(report['macro avg']['support'])
metrics['Acurácia'].append(report['accuracy'])  # Acurácia na coluna "Média"

# Adicionar as métricas de média ponderada
metrics['Precisão'].append(report['weighted avg']['precision'])
metrics['Recall'].append(report['weighted avg']['recall'])
metrics['F1-Score'].append(report['weighted avg']['f1-score'])
metrics['Suporte'].append(None)  # Suporte não tem média ponderada, então deixamos como None
metrics['Acurácia'].append(None)  # Acurácia não é aplicável à média ponderada

# Criar DataFrame com as métricas
index = list(present_genres) + ['Média', 'Média Ponderada']
metrics_df = pd.DataFrame(metrics, index=index)

# Transpor o DataFrame para que as métricas fiquem nas linhas e as classes nas colunas
metrics_df = metrics_df.transpose()

# Exibir a tabela
st.write("Métricas de Avaliação:")
st.dataframe(metrics_df)

st.markdown("""
- **Precision**: A proporção de previsões corretas para um gênero específico.
- **Recall**: A proporção de instâncias corretamente classificadas de um gênero.
- **F1-Score**: A média harmônica entre precision e recall.
- **Support**: O número de ocorrências de cada gênero no conjunto de teste.
- **Acurácia**: A proporção de previsões corretas em relação ao total de previsões.
""")

# =====================================
# Explicabilidade com SHAP
# =====================================
st.write("### 🔍 Explicabilidade do Modelo com SHAP")

st.markdown("""
O SHAP (SHapley Additive exPlanations) é um método baseado na teoria dos valores de Shapley, que vem da teoria dos jogos. 
Ele permite entender **quais características mais influenciam as previsões do modelo**, mostrando a importância de cada palavra-chave na determinação do gênero musical.

Aqui, usamos o SHAP para visualizar como o modelo Random Forest classifica os gêneros musicais. O gráfico gerado exibe quais palavras-chave têm maior impacto na decisão do modelo.
""")

# Checkbox para ativar a explicabilidade SHAP
explain_shap = st.checkbox("Gerar Explicabilidade SHAP (pode ser lento)")

if explain_shap and model_choice == "Random Forest":
    st.markdown("""
    O gráfico abaixo mostra a contribuição de cada palavra-chave para a decisão do modelo. 
    - **Cores**: Representam diferentes classes (gêneros musicais).
    - **Barras maiores**: Indicam que a característica teve um impacto significativo na previsão do modelo.
    """)

    # Criando o explicador SHAP para o modelo Random Forest
    explainer = shap.Explainer(model, X_train_resampled)
    shap_values = explainer(X_test[:50])  # Pegamos apenas 50 amostras para otimizar o tempo de geração
    
    # Criando a figura antes de chamar o SHAP
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test[:50], feature_names=X_test.columns, show=False)
    
    # Exibindo o gráfico no Streamlit
    st.pyplot(fig)

    st.markdown("""
    🔹 **Interpretação do gráfico**:
    - Se uma palavra-chave aparece frequentemente no topo, significa que ela influencia fortemente as previsões do modelo.
    - O tamanho da barra indica a magnitude do impacto da palavra-chave no resultado final.
    - As cores mostram a contribuição para diferentes classes de gêneros musicais.

    Este tipo de análise ajuda a entender **como** o modelo toma suas decisões e **se ele está aprendendo corretamente os padrões das letras musicais**.
    """)


# =====================================
# Seção 3: Simulação de Classificação
# =====================================
st.write("### 🎵 Simulação de Classificação")
st.markdown("""
Insira palavras-chave ou uma letra de música para classificar o gênero.
""")

# Input do usuário
user_input = st.text_area("Insira palavras-chave ou uma letra de música:")

if user_input:
    user_counts = {genre_name: count_keywords(user_input, genre_keywords) for genre_name, genre_keywords in keywords.items()}
    user_X = pd.DataFrame([list(user_counts.values())], columns=keywords.keys())

    # Prever o gênero e as probabilidades
    predicted_genre = model.predict(user_X)
    predicted_proba = model.predict_proba(user_X)

    st.success(f"### 🎶 O gênero previsto é: **{predicted_genre[0]}**")
    
    # Exibir probabilidades
    st.write("### Probabilidades por Gênero:")
    proba_df = pd.DataFrame(predicted_proba, columns=model.classes_)
    st.dataframe(proba_df)

    # Gráfico de barras das probabilidades
    st.write("### 📊 Probabilidades por Gênero (Gráfico)")
    plt.figure(figsize=(10, 6))
    proba_df.T.plot(kind='barh', legend=False)  # Gráfico horizontal
    plt.xlabel("Probabilidade")
    plt.ylabel("Gênero Musical")
    plt.title("Probabilidades de Classificação por Gênero")
    st.pyplot(plt)

class_distribution = df["genre"].value_counts()
print("Distribuição das Classes:")
print(class_distribution)

# Crie um gráfico de barras para visualizar a distribuição
plt.figure(figsize=(10, 6))
plt.bar(class_distribution.index, class_distribution.values, color='skyblue')
plt.xlabel("Gênero Musical")
plt.ylabel("Número de Exemplos")
plt.title("Distribuição das Classes no Conjunto de Treinamento")
plt.xticks(rotation=45)  # Rotaciona os rótulos do eixo X para melhor visualização
plt.show()