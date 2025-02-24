import streamlit as st

st.set_page_config(page_title="Sintonize", layout="wide")
visao_geral = st.Page("overview.py", title="Visão Geral", icon=":material/home:", default=True)
distribuicao = st.Page("decades.py", title="Distribuição por Décadas", icon=":material/query_stats:")
evolucao = st.Page("acoustic.py", title="Evolução Acústica", icon=":material/chart_data:")
outliers = st.Page("outliers.py", title="Análise de Outliers", icon=":material/outlined_flag:")
palavras_chave = st.Page("keywords.py", title="Palavras-chave e Contexto Histórico", icon=":material/key:")
clusterizacao = st.Page("clustering.py", title="Clusterização", icon=":material/grain:")
classificacao = st.Page("genre_classification.py", title="Classificação de Gêneros", icon=":material/library_music:")
previsao = st.Page("prediction.py", title="Previsão de Tendências de Gêneros", icon=":material/search_insights:")
pg = st.navigation(
    {
        "Análises": [visao_geral, distribuicao, evolucao, outliers],
        "Modelos": [palavras_chave, clusterizacao, classificacao, previsao],
    }
)
pg.run()