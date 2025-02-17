import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from utils import load_data

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

# Página de Previsão de Tendências de Gêneros
st.title("📈 Previsão de Tendências de Gêneros")
st.markdown("""
A análise a seguir utiliza **Random Forest Regressor** para prever a popularidade dos gêneros musicais ao longo do tempo.
""")

# Agrupar os dados por ano e gênero
genre_trends = df.groupby(['year', 'genre']).size().reset_index(name='count')

# Selecionar um gênero para análise
selected_genre = st.selectbox("🎼 Escolha um gênero para previsão:", genre_trends['genre'].unique())

# Filtrar os dados pelo gênero selecionado
filtered_data = genre_trends[genre_trends['genre'] == selected_genre]

# Criar variáveis para treinamento
X = filtered_data[['year']]
y = filtered_data['count']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Previsão para os próximos anos
future_years = np.array(range(int(df['year'].max()) + 1, int(df['year'].max()) + 11)).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Avaliação do modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"📊 **Erro Médio Absoluto (MAE):** {mae:.2f}")

# Criar gráfico da previsão
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(filtered_data['year'], filtered_data['count'], label="Dados Históricos", marker='o')
ax.plot(future_years, future_predictions, label="Previsão", linestyle='dashed', marker='o', color='red')
ax.set_xlabel("Ano")
ax.set_ylabel("Popularidade")
ax.set_title(f"📈 Tendência do gênero {selected_genre}")
ax.legend()
st.pyplot(fig)
