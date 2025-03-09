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

# Garantir que a coluna 'year' esteja presente
if 'year' not in df.columns:
    if 'release_date' in df.columns:
        df['release_date'] = df['release_date'].astype(str).str.replace(",", "").str.split(".").str[0]
        df['year'] = pd.to_numeric(df['release_date'], errors='coerce')
        df['decade'] = (df['year'] // 10 * 10).astype('Int64')
    else:
        st.error("Erro: A coluna 'year' não foi encontrada no dataset.")
        st.stop()

# -------------------------------------------------
#  📈 SEÇÃO 1 - PREVISÃO DE POPULARIDADE DE GÊNEROS (PP1)
# -------------------------------------------------
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

# -------------------------------------------------
#  🔮 SEÇÃO 2 - PROJEÇÃO TEMPORAL DE ATRIBUTOS MUSICAIS (PP2)
# -------------------------------------------------
st.title("🔮 Previsão da Evolução Musical")
st.markdown("""
Nesta seção, analisamos tendências musicais ao longo das décadas e prevemos suas mudanças futuras até **2050**.
""")

# Selecionar atributos para previsão
attributes = ['danceability', 'energy', 'valence', 'loudness']
selected_attribute = st.selectbox("🎵 Escolha um atributo para prever sua evolução:", attributes)

# Agrupar os dados por década e calcular a média do atributo selecionado
df_grouped = df.groupby("decade")[selected_attribute].mean().reset_index()

# Criar variáveis para previsão
X_attr = df_grouped["decade"].values.reshape(-1, 1)
y_attr = df_grouped[selected_attribute].values

# Dividir os dados em treino e teste
X_train_attr, X_test_attr, y_train_attr, y_test_attr = train_test_split(X_attr, y_attr, test_size=0.2, random_state=42)

# Criar e treinar o modelo de Random Forest
model_attr = RandomForestRegressor(n_estimators=100, random_state=42)
model_attr.fit(X_train_attr, y_train_attr)

# Prever valores futuros (2030, 2040, 2050)
future_decades = np.array([2030, 2040, 2050]).reshape(-1, 1)
future_attr_predictions = model_attr.predict(future_decades)

# Avaliação do modelo
y_pred_attr = model_attr.predict(X_test_attr)
mae_attr = mean_absolute_error(y_test_attr, y_pred_attr)
st.write(f"📊 **Erro Médio Absoluto (MAE) para {selected_attribute}:** {mae_attr:.2f}")

# Criar DataFrame com previsões futuras
forecast_df_attr = pd.DataFrame({
    "Década": [2030, 2040, 2050],
    selected_attribute.capitalize(): future_attr_predictions
})

# Exibir tabela de previsões
st.write("### 📊 Projeção de Valores Futuros")
st.dataframe(forecast_df_attr)

# Criar gráfico da previsão
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_grouped["decade"], df_grouped[selected_attribute], label="Dados Históricos", marker='o')
ax.plot(future_decades, future_attr_predictions, label="Previsão", linestyle='dashed', marker='o', color='red')
ax.set_xlabel("Década")
ax.set_ylabel(selected_attribute.capitalize())
ax.set_title(f"📈 Evolução de {selected_attribute.capitalize()} até 2050")
ax.legend()
st.pyplot(fig)

st.markdown("""
Os modelos indicam que **o atributo selecionado** continuará seguindo uma tendência baseada no comportamento passado. 
A análise sugere mudanças na estrutura da música popular nos próximos anos.
""")

st.write("#### Como isso responde à PP2:")
st.write("##### _Como as tendências lineares da evolução musical podem representar o futuro da sociedade?_")

st.markdown("""
- Permite comparar diferentes abordagens para prever a música do futuro.
- Mostra como as tendências musicais evoluem e impactam a sociedade.
- Ilustra o efeito das tecnologias na personalização musical.
""")