from util.util_db import consultar_dados, credenciais_banco
from modelos.modelo_associacao import modelo_associacao
from modelos.modelo_clusters import modelo_clusters
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get('/api/relatorio-cluster')
def relatorio_cluster():
    """
    Endpoint para obter os dados de clusterização.
    """
    try:
        df_cluster = modelo_clusters()
        # Converte o DataFrame para um formato JSON
        resultado = df_cluster.to_dict(orient='records')
        return resultado
    except Exception as e:
        return {"erro": str(e)}

@app.get('/api/relatorio-associacao')
def relatorio_associacao():
    """
    Endpoint para obter as regras de associação.
    """
    try:
        dic_credenciais = credenciais_banco()
        url = dic_credenciais["url_banco"]
        key = dic_credenciais["key_banco"]

        # Define a tabela e colunas para o modelo de associação
        tabela = "tbl_arq4_2021"
        colunas = ["QE_I57", "QE_I30", "QE_I56"]

        # Busca e prepara os dados
        df_dados = consultar_dados(tabela, url, key)
        df_filtrado = df_dados[colunas]

        # Gera as regras de associação
        regras = modelo_associacao(df_filtrado, suporte_minimo=0.05, confianca_minima=0.5)

        # Converte os frozensets para listas para serem serializáveis em JSON
        regras['antecedents'] = regras['antecedents'].apply(list)
        regras['consequents'] = regras['consequents'].apply(list)

        resultado = regras.to_dict(orient='records')
        return resultado
    except Exception as e:
        return {"erro": str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
