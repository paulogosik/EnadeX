from supabase import create_client, Client
from pandas import DataFrame
import pandas as pd
import dotenv
import os

def credenciais_banco() -> dict[str,str]:
    dotenv.load_dotenv()
    prog_id = os.getenv("SUPABASE_URL")
    url_banco = f"https://{prog_id}.supabase.co"
    key_banco = os.getenv("SUPABASE_KEY")
    return {"url_banco": url_banco, "key_banco": key_banco}

def consultar_dados(
        table_name: str,
        supabase_url: str,
        supabase_key: str,
        colunas: str = "*",
        filtros: dict = None,
        limite: int = 1000,
        dic_tipagem: dict = None
) -> DataFrame:
    """
    Realiza consultas no Supabase e retorna os dados diretamente em um DataFrame do Pandas
    com as tipagens devidamente corrigidas.

    :param table_name: Nome da tabela a ser consultada.
    :param supabase_url: URL do seu projeto Supabase.
    :param supabase_key: Chave de API do Supabase.
    :param colunas: Colunas desejadas (ex: "id, CO_CURSO"). O padrão é "*" (todas).
    :param filtros: Dicionário para correspondências exatas (ex: {"CO_CURSO": "15002"}).
    :param limite: Quantidade máxima de registros a retornar.
    :param dic_tipagem: O nosso dicionário de mapeamento de tipos (opcional).
    :return: DataFrame do Pandas com os resultados da consulta.
    """
    # 1. Inicializa o cliente
    supabase: Client = create_client(supabase_url, supabase_key)

    try:
        # 2. Constrói e executa a query no Supabase
        query = supabase.table(table_name).select(colunas)

        if filtros:
            for coluna, valor in filtros.items():
                query = query.eq(coluna, valor)

        query = query.limit(limite)
        resposta = query.execute()

        # 3. Extrai a lista de registros da resposta
        registros = resposta.data if hasattr(resposta, 'data') else resposta.get('data', [])

        # 4. Transforma diretamente em um DataFrame do Pandas
        df_resultado = pd.DataFrame(registros)

        # 5. Se o DataFrame não estiver vazio e o dicionário de tipos for passado, aplica a tipagem técnica
        if not df_resultado.empty and dic_tipagem:
            # Filtra o dicionário para aplicar apenas nas colunas que realmente vieram na consulta
            tipagem_valida = {col: tipo for col, tipo in dic_tipagem.items() if col in df_resultado.columns}
            df_resultado = df_resultado.astype(tipagem_valida)

        return df_resultado

    except Exception as e:
        print(f"Falha ao realizar a consulta e converter para DataFrame na tabela '{table_name}': {e}")
        raise e