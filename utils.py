import pandas as pd
import streamlit as st

def load_data(path):
    """Carrega os dados do arquivo Parquet."""
    try:
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

def count_keywords(text, keywords_list):
    """Conta a ocorrência de palavras-chave em um texto."""
    count = 0
    for word in keywords_list:
        count += text.lower().count(word.lower())
    return count
