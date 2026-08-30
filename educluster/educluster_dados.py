import pandas as pd

from educluster.educluster_config import (
    COLS_ARQ3_CURSO,
    COLS_ARQ4_CURSO,
    DIR_CACHE,
    DIR_MICRODADOS,
    NOTAS_ARQ3,
    ITENS_ARQ4,
    TABELAS_SUPABASE,
)

SEPARADOR = ";"
CODIFICACAO = "latin-1"


def _caminho_local(numero_arq: int):
    return DIR_MICRODADOS / f"microdados2021_arq{numero_arq}.txt"


def _caminho_cache(numero_arq: int):
    return DIR_CACHE / f"arq{numero_arq}.parquet"


def _converter_numericos(df: pd.DataFrame, colunas: list) -> pd.DataFrame:
    for coluna in colunas:
        if coluna in df.columns:
            serie = df[coluna]
            if serie.dtype == object or isinstance(serie.dtype, pd.StringDtype):
                serie = serie.str.replace(",", ".", regex=False)
            df[coluna] = pd.to_numeric(serie, errors="coerce")
    return df


def carregar_local(numero_arq: int, colunas: list = None, usar_cache: bool = True) -> pd.DataFrame:
    cache = _caminho_cache(numero_arq)
    if usar_cache and cache.exists():
        df = pd.read_parquet(cache)
        return df[colunas].copy() if colunas else df

    caminho = _caminho_local(numero_arq)
    if not caminho.exists():
        raise FileNotFoundError(f"Microdado nao encontrado: {caminho}")

    print(f"[educluster] lendo {caminho.name} do disco")
    df = pd.read_csv(caminho, sep=SEPARADOR, encoding=CODIFICACAO, dtype=str)
    df = _converter_numericos(df, ["CO_CURSO"] + NOTAS_ARQ3 + ITENS_ARQ4)

    if usar_cache:
        DIR_CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        print(f"[educluster] cache gravado em {cache.name}")

    return df[colunas].copy() if colunas else df


def carregar_supabase(numero_arq: int, colunas: list = None) -> pd.DataFrame:
    from util.util_db import consultar_dados, credenciais_banco

    tabela = TABELAS_SUPABASE[numero_arq]
    credenciais = credenciais_banco()
    projecao = ",".join(colunas) if colunas else "*"

    print(f"[educluster] consultando {tabela} no Supabase")
    df = consultar_dados(tabela, credenciais["url_banco"], credenciais["key_banco"], colunas=projecao)
    return _converter_numericos(df, ["CO_CURSO"] + NOTAS_ARQ3 + ITENS_ARQ4)


def carregar_arquivo(numero_arq: int, colunas: list = None, origem: str = "local") -> pd.DataFrame:
    if origem == "local":
        return carregar_local(numero_arq, colunas)
    if origem == "supabase":
        return carregar_supabase(numero_arq, colunas)
    raise ValueError(f"origem invalida: {origem}. Use 'local' ou 'supabase'.")


def carregar_arq3(origem: str = "local") -> pd.DataFrame:
    return carregar_arquivo(3, COLS_ARQ3_CURSO, origem)


def carregar_arq4(origem: str = "local") -> pd.DataFrame:
    return carregar_arquivo(4, COLS_ARQ4_CURSO, origem)
