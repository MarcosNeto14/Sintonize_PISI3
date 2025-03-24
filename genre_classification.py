import streamlit as st
import pandas as pd
import shap
import numpy as np
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
    "Selecione o modelo de classificação (caso queira analisar com o método SHAP escolha Random Forest):",
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
# Seção 3: Simulação de Classificação
# =====================================
st.write("### 🎵 Simulação de Classificação")
st.markdown("""
Insira palavras-chave ou uma letra de música para classificar o gênero.
""")

# Input do usuário
user_input = st.text_area("Insira palavras-chave ou uma letra de música (separadas por espaço ou vírgula):")

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
    # =====================================
    # Explicabilidade com SHAP
    # =====================================
    # Ativando o SHAP apenas para Random Forest
    if model_choice == "Random Forest":
        st.write("### 🧠 Explicação da decisão do modelo (SHAP)")
        explain_user_shap = st.checkbox("Mostrar explicação SHAP para essa previsão")

        if explain_user_shap:
            with st.spinner("Calculando explicação SHAP..."):
                explainer = shap.Explainer(model, X_train_resampled)
                shap_values = explainer(user_X)
                predicted_index = np.argmax(predicted_proba)

                # Reconstruir explicação individual da classe prevista
                user_shap = shap.Explanation(
                    values=shap_values.values[0][predicted_index],
                    base_values=shap_values.base_values[0][predicted_index],
                    data=user_X.values[0],
                    feature_names=user_X.columns.tolist()
                )

                fig, ax = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(user_shap, show=False)
                st.pyplot(fig)

            st.markdown("""
                #### ✨ Sobre a explicabilidade com SHAP

                O gráfico abaixo utiliza o método **SHAP (SHapley Additive exPlanations)**, baseado na teoria dos jogos, para explicar como o modelo chegou à decisão de gênero para a letra fornecida.

                O **SHAP calcula o impacto de cada característica (nesse caso, palavras-chave)** na previsão feita pelo modelo. Ele mostra quais palavras **empurraram** o modelo para escolher aquele gênero e quais tentaram puxar para outro.

                #### 🧠 Como interpretar o gráfico:

                - A **barra azul** representa a base média de decisão do modelo (o "ponto neutro").
                - As **setas vermelhas** indicam palavras que contribuíram positivamente para o gênero previsto.
                - As **setas azuis** mostram palavras que puxaram contra esse gênero.
                - Quanto mais **no topo**, mais importante foi a palavra na decisão final.
                - O **tamanho da barra** indica o peso que a palavra teve no resultado.

                > Isso permite entender o comportamento do modelo de forma transparente, e também ajuda a verificar se ele está aprendendo padrões coerentes das letras musicais.
                """)
            
            st.markdown("#### 📈 Importância Global das Palavras-chave")
            st.markdown("Cada barra representa o impacto da contagem de palavras-chaves relacionadas ao gênero selecionado na predição final.")

            with st.spinner("Gerando gráfico geral de importância das features..."):
                # Escolhe um subconjunto dos dados de teste
                X_sample = X_test[:100]

                # Cria o explicador
                explainer = shap.Explainer(model, X_train_resampled)
                shap_values_sample = explainer(X_sample)

                # Se o modelo for multi-classe, vamos pegar a classe 0 (ou qualquer uma)
                if hasattr(shap_values_sample, "values") and len(shap_values_sample.values.shape) == 3:
                    # Garante que estamos pegando a estrutura Explanation certa
                    shap_values_bar = shap.Explanation(
                        values=shap_values_sample.values[:, 0],
                        base_values=shap_values_sample.base_values[:, 0],
                        data=X_sample.values,
                        feature_names=X_sample.columns.tolist()
                    )
                else:
                    shap_values_bar = shap_values_sample

                # Exibir gráfico
                fig_bar, ax = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values_bar, show=False)
                st.pyplot(fig_bar)
                
            st.markdown("""
            #### 🧠 O que o gráfico de barras SHAP está mostrando?

            Esse gráfico apresenta uma visão **global** da importância das palavras-chave (features) utilizadas pelo modelo de classificação.

            Em vez de explicar **uma única previsão** como o gráfico anterior (waterfall), aqui o SHAP analisa um conjunto de exemplos (várias músicas) e calcula, em média, **quais palavras mais influenciam as decisões do modelo**.

            ##### ✨ Como interpretar:

            - Cada barra representa uma **palavra-chave** (feature).
            - O comprimento da barra indica **a força média do impacto** dessa palavra nas decisões do modelo.
            - Quanto maior a barra, mais aquela palavra influenciou as classificações em geral.
            - Esse gráfico é útil para entender **o que o modelo realmente está aprendendo** e valorizando nas letras das músicas.

            > Exemplo: Se a palavra "love" tiver uma barra longa, quer dizer que ela aparece com frequência nas letras rotuladas como "pop", e o modelo aprendeu isso direitinho.

            Essa explicação ajuda a garantir que o modelo esteja tomando decisões coerentes com o esperado e pode ser usada até como **instrumento de análise de padrões culturais** ao longo do tempo.
            """)


            




            



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