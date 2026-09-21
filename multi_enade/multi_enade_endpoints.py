from fastapi import APIRouter
from util.util_db import consultar_dados, credenciais_banco

router = APIRouter(prefix="/api/multienade", tags=["MultiENADE"])

@router.get("/relatorio-regressao")
def multi_enade_relatorio_regressao():
    try:
        dic_credenciais = credenciais_banco()
        df_regressao = consultar_dados(
            "tbl_multi_enade_regressao",
            dic_credenciais["url_banco"],
            dic_credenciais["key_banco"],
        )
        return df_regressao.to_dict(orient="records")
    except Exception as e:
        return {"erro": f"Falha ao consultar regressão: {str(e)}"}

@router.get("/relatorio-cluster")
def multi_enade_relatorio_cluster():
    try:
        dic_credenciais = credenciais_banco()
        df_cluster = consultar_dados(
            "tbl_multi_enade_clusters",
            dic_credenciais["url_banco"],
            dic_credenciais["key_banco"],
        )
        return df_cluster.to_dict(orient="records")
    except Exception as e:
        return {"erro": f"Falha ao consultar clusters: {str(e)}"}

@router.get("/relatorio-associacao")
def multi_enade_relatorio_associacao():
    try:
        dic_credenciais = credenciais_banco()
        df_regras = consultar_dados(
            "tbl_multi_enade_associacao",
            dic_credenciais["url_banco"],
            dic_credenciais["key_banco"],
        )
        return df_regras.to_dict(orient="records")
    except Exception as e:
        return {"erro": f"Falha ao consultar associação: {str(e)}"}