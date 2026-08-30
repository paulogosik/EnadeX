from util.util_db import consultar_dados, credenciais_banco
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get('/api/relatorio-regressao')
def multi_enade_relatorio_regressao():
    """
    Endpoint para obter a matriz de dados com as notas previstas pelo Random Forest.
    """
    try:
        dic_credenciais = credenciais_banco()
        # Lê a tabela do banco onde o script modelo_regressao.py fez o upsert
        df_regressao = consultar_dados("tbl_multi_enade_regressao", dic_credenciais["url_banco"], dic_credenciais["key_banco"])
        return df_regressao.to_dict(orient='records')
    except Exception as e:
        return {"erro": f"Falha ao consultar regressão: {str(e)}"}

@app.get('/api/relatorio-cluster')
def multi_enade_relatorio_cluster():
    """
    Endpoint para obter os dados de clusterização e as coordenadas PCA.
    """
    try:
        dic_credenciais = credenciais_banco()
        # Lê a tabela do banco onde o script modelo_clusters.py fez o upsert
        df_cluster = consultar_dados("tbl_multi_enade_clusters", dic_credenciais["url_banco"], dic_credenciais["key_banco"])
        return df_cluster.to_dict(orient='records')
    except Exception as e:
        return {"erro": f"Falha ao consultar clusters: {str(e)}"}

@app.get('/api/relatorio-associacao')
def multi_enade_relatorio_associacao():
    """
    Endpoint para obter as regras de associação (Apriori).
    """
    try:
        dic_credenciais = credenciais_banco()
        # Lê a tabela do banco que acabamos de configurar com o ID (tbl_multi_enade_associacao)
        df_regras = consultar_dados("tbl_multi_enade_associacao", dic_credenciais["url_banco"], dic_credenciais["key_banco"])
        return df_regras.to_dict(orient='records')
    except Exception as e:
        return {"erro": f"Falha ao consultar associação: {str(e)}"}

if __name__ == '__main__':
    import uvicorn
    # Rodando na porta 8000 para que o Streamlit ou outro frontend consuma facilmente
    uvicorn.run(app, host="0.0.0.0", port=8000)