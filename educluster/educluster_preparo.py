import pandas as pd

from educluster.educluster_config import (
    CODIGOS_NAO_RESPOSTA_ARQ4,
    CODIGO_PRESENCA_VALIDA,
    DIMENSOES_ARQ4,
    ESCALA_ARQ4_MAXIMA,
    ESCALA_ARQ4_MINIMA,
    ITENS_ARQ4,
    N_MINIMO_RESPONDENTES,
)


def limpar_escala_arq4(df_arq4: pd.DataFrame) -> pd.DataFrame:
    df = df_arq4.copy()
    itens = [c for c in ITENS_ARQ4 if c in df.columns]
    bloco = df[itens]
    fora_da_escala = (
        bloco.isin(CODIGOS_NAO_RESPOSTA_ARQ4)
        | (bloco < ESCALA_ARQ4_MINIMA)
        | (bloco > ESCALA_ARQ4_MAXIMA)
    )
    df[itens] = bloco.mask(fora_da_escala)
    return df


def calcular_dimensoes_arq4(df_arq4: pd.DataFrame) -> pd.DataFrame:
    df = df_arq4.copy()
    for dimensao, itens in DIMENSOES_ARQ4.items():
        presentes = [c for c in itens if c in df.columns]
        df[dimensao] = df[presentes].mean(axis=1)
    return df


def derivar_area(df_arq3: pd.DataFrame) -> pd.DataFrame:
    df = df_arq3.copy()
    gabaritos = sorted(df["DS_VT_GAB_OCE_FIN"].dropna().unique())
    mapa = {gab: f"A{i:02d}" for i, gab in enumerate(gabaritos, start=1)}
    df["area"] = df["DS_VT_GAB_OCE_FIN"].map(mapa)
    return df


def agregar_arq3_por_curso(df_arq3: pd.DataFrame) -> pd.DataFrame:
    df = derivar_area(df_arq3)
    df["TP_PRES"] = df["TP_PRES"].astype(str)

    presenca = (
        df.groupby("CO_CURSO")["TP_PRES"]
        .apply(lambda s: (s == CODIGO_PRESENCA_VALIDA).mean())
        .rename("tx_presenca")
    )

    com_nota = df[df["NT_GER"].notna()]
    agregado = com_nota.groupby("CO_CURSO").agg(
        n_notas=("NT_GER", "size"),
        area=("area", "first"),
        NT_GER=("NT_GER", "mean"),
        NT_FG=("NT_FG", "mean"),
        NT_CE=("NT_CE", "mean"),
        NT_OBJ_FG=("NT_OBJ_FG", "mean"),
        NT_DIS_FG=("NT_DIS_FG", "mean"),
        NT_OBJ_CE=("NT_OBJ_CE", "mean"),
        NT_DIS_CE=("NT_DIS_CE", "mean"),
    )
    return agregado.join(presenca)


def agregar_arq4_por_curso(df_arq4: pd.DataFrame) -> pd.DataFrame:
    df = calcular_dimensoes_arq4(limpar_escala_arq4(df_arq4))
    dimensoes = list(DIMENSOES_ARQ4.keys())
    respondentes = df[df[dimensoes].notna().any(axis=1)]
    agregado = respondentes.groupby("CO_CURSO")[dimensoes].mean()
    agregado["n_qe"] = respondentes.groupby("CO_CURSO").size()
    return agregado


def padronizar_por_area(df_curso: pd.DataFrame, colunas: list) -> pd.DataFrame:
    df = df_curso.copy()
    for coluna in colunas:
        grupo = df.groupby("area")[coluna]
        desvio = grupo.transform("std")
        df[f"{coluna}_z"] = (df[coluna] - grupo.transform("mean")) / desvio.replace(0, pd.NA)
    return df


def montar_base_curso(
    df_arq3: pd.DataFrame,
    df_arq4: pd.DataFrame,
    n_minimo: int = N_MINIMO_RESPONDENTES,
) -> pd.DataFrame:
    lado_notas = agregar_arq3_por_curso(df_arq3)
    lado_percepcao = agregar_arq4_por_curso(df_arq4)

    base = lado_notas.join(lado_percepcao, how="inner")
    total_bruto = len(base)

    base = base[(base["n_notas"] >= n_minimo) & (base["n_qe"] >= n_minimo)]
    base = base.dropna(subset=list(DIMENSOES_ARQ4.keys()) + ["NT_FG", "NT_CE", "area"])

    print(
        f"[educluster] base de curso: {total_bruto} cursos apos join, "
        f"{len(base)} com n >= {n_minimo} nos dois arquivos "
        f"({int(base['n_notas'].sum())} estudantes, {base['area'].nunique()} areas)"
    )
    return base
