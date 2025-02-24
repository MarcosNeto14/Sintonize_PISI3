import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

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

st.title("📊 Análise de Outliers")
st.markdown("""
A análise a seguir identifica e remove outliers dos atributos acústicos do dataset. 
Outliers são pontos de dados que se desviam significativamente da distribuição normal e podem distorcer análises e modelos.
""")

attributes = ["danceability", "energy", "valence", "loudness"]
selected_attributes = st.multiselect(
    "Selecione os atributos para análise de outliers:",
    options=attributes,
    default=attributes
)

def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

if selected_attributes:
    st.write("### Identificação de Outliers")
    for attribute in selected_attributes:
        st.write(f"#### Atributo: {attribute.capitalize()}")
        
        outliers = detect_outliers(df, attribute)
        st.write(f"Número de outliers encontrados: **{len(outliers)}**")
        
        st.write("**Boxplot antes da remoção de outliers:**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df[attribute], ax=ax)
        st.pyplot(fig)
        
        df_cleaned = df[~df.index.isin(outliers.index)]
        
        st.write("**Boxplot após a remoção de outliers:**")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(x=df_cleaned[attribute], ax=ax)
        st.pyplot(fig)
        
        df = df_cleaned

    st.success("Outliers removidos com sucesso! O dataset foi atualizado.")
else:
    st.warning("Selecione pelo menos um atributo para análise.")