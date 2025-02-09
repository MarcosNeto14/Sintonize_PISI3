import streamlit as st

# configuração inicial do Streamlit
st.set_page_config(page_title="Sintonize", layout="wide")

# criando uma barra de navegação com categorias

## área de análises
visao_geral = st.Page("overview.py", title="Visão Geral", icon=":material/home:", default=True)
distribuicao = st.Page("decades.py", title="Distribuição por Décadas", icon=":material/query_stats:")
evolucao = st.Page("acoustic.py", title="Evolução Acústica", icon=":material/chart_data:")

## área de modelos 
palavras_chave = st.Page("keywords.py", title="Palavras-chave e Contexto Histórico", icon=":material/key:")
clusterizacao = st.Page("clustering.py", title="Clusterização", icon=":material/grain:")
classificacao = st.Page("classification.py", title="Classificação de Gêneros", icon=":material/library_music:")
previsao = st.Page("prediction.py", title="Previsão de Tendências de Gêneros", icon=":material/search_insights:")

# chamando o sidemenu
pg = st.navigation(
    {
        "Análises": [visao_geral, distribuicao, evolucao],
        "Modelos": [palavras_chave, clusterizacao, classificacao, previsao],
    }
)

# rodando
pg.run()