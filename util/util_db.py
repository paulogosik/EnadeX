from supabase import create_client, Client
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from pandas import DataFrame
import pandas as pd
import numpy as np
import warnings
import dotenv
import os

# Silenciando avisos desnecessários para manter seu console limpo
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")


def credenciais_banco() -> dict[str, str]:
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
        limite: int = None,
        dic_tipagem: dict = None
) -> DataFrame:
    """
    Realiza consultas no Supabase com paginação automática para burlar
    o limite de 1000 registros, retornando os dados em um DataFrame.
    """
    supabase: Client = create_client(supabase_url, supabase_key)

    try:
        todos_registros = []
        tamanho_pagina = 1000
        offset = 0

        while True:
            query = supabase.table(table_name).select(colunas)

            if filtros:
                for coluna, valor in filtros.items():
                    query = query.eq(coluna, valor)

            fim = offset + tamanho_pagina - 1

            if limite:
                if offset >= limite:
                    break
                if fim >= limite:
                    fim = limite - 1

            resposta = query.range(offset, fim).execute()
            registros_pagina = resposta.data if hasattr(resposta, 'data') else resposta.get('data', [])
            todos_registros.extend(registros_pagina)

            if len(registros_pagina) < tamanho_pagina or (limite and len(todos_registros) >= limite):
                break

            offset += tamanho_pagina

        df_resultado = pd.DataFrame(todos_registros)

        if not df_resultado.empty and dic_tipagem:
            tipagem_valida = {col: tipo for col, tipo in dic_tipagem.items() if col in df_resultado.columns}
            df_resultado = df_resultado.astype(tipagem_valida)

        return df_resultado

    except Exception as e:
        print(f"Falha ao realizar a consulta e converter para DataFrame na tabela '{table_name}': {e}")
        raise e


def preparar_dados_regressao(df_arq3: DataFrame, df_arq4: DataFrame, df_arq21: DataFrame,
                             df_arq29: DataFrame) -> DataFrame:
    """
    Trata as bases de dados e aplica engenharia de atributos (feature engineering)
    para consolidar o perfil acadêmico e socioeconômico completo do curso.
    """
    # 1. Agregação do Arquivo 3 (Notas de Desempenho e Dados Socioeconômicos)
    df_arq3_limpo = df_arq3[['CO_CURSO', 'NT_GER']].copy()
    df_arq3_limpo['NT_GER'] = pd.to_numeric(df_arq3_limpo['NT_GER'], errors='coerce')
    df_arq3_limpo = df_arq3_limpo[(df_arq3_limpo['NT_GER'].notna()) & (df_arq3_limpo['NT_GER'] > 0)]
    df_arq3_agg = df_arq3_limpo.groupby('CO_CURSO')['NT_GER'].mean().reset_index()

    # Extraindo as variáveis socioeconômicas (Renda, Escolaridade, etc.) do ARQ3 se existirem
    cols_socioeconomicas = [col for col in df_arq3.columns if col.startswith('CO_RS_I')]
    if cols_socioeconomicas:
        df_socio = df_arq3[['CO_CURSO'] + cols_socioeconomicas].copy()
        # Converte letras em proporções (dummies) para o curso
        df_socio_dummies = pd.get_dummies(df_socio, columns=cols_socioeconomicas, dtype=int)
        df_socio_agg = df_socio_dummies.groupby('CO_CURSO').mean().reset_index()
        df_arq3_agg = df_arq3_agg.merge(df_socio_agg, on='CO_CURSO', how='left')

    # 2. Transformação Dinâmica do Arquivo 4 (Percepção do Processo Formativo)
    # Pega dinamicamente todas as colunas de avaliação I27 a I68
    cols_avaliacao = [col for col in df_arq4.columns if col.startswith('QE_I')]
    df_arq4_limpo = df_arq4[['CO_CURSO'] + cols_avaliacao].copy()

    for col in cols_avaliacao:
        df_arq4_limpo[col] = pd.to_numeric(df_arq4_limpo[col], errors='coerce')
        # Remove valores que quebram a escala hierárquica (6=Não sei, 7/8=Não se aplica)
        df_arq4_limpo = df_arq4_limpo[~df_arq4_limpo[col].isin([6, 7, 8])]

    df_arq4_agg = df_arq4_limpo.groupby('CO_CURSO')[cols_avaliacao].mean().reset_index()

    # 3. Transformação e agregação do Arquivo 21 (Ações Afirmativas e Cotas)
    df_arq21_limpo = df_arq21[['CO_CURSO', 'QE_I15']].dropna().copy()
    df_arq21_limpo['QE_I15'] = df_arq21_limpo['QE_I15'].astype(str).str.strip().str.upper()
    df_arq21_dummies = pd.get_dummies(df_arq21_limpo, columns=['QE_I15'], dtype=int)
    df_arq21_agg = df_arq21_dummies.groupby('CO_CURSO').mean().reset_index()

    # 4. Transformação e agregação do Arquivo 29 (Horas de Dedicação aos Estudos)
    df_arq29_limpo = df_arq29[['CO_CURSO', 'QE_I23']].dropna().copy()
    df_arq29_limpo['QE_I23'] = df_arq29_limpo['QE_I23'].astype(str).str.strip().str.upper()
    map_horas = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df_arq29_limpo['QE_I23_ordinal'] = df_arq29_limpo['QE_I23'].map(map_horas)
    df_arq29_agg = df_arq29_limpo.dropna(subset=['QE_I23_ordinal']).groupby('CO_CURSO')[
        'QE_I23_ordinal'].mean().reset_index()

    # 5. Consolidação final das tabelas (Inner Join para manter consistência absoluta)
    df_consolidado = df_arq3_agg.merge(df_arq4_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq21_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq29_agg, on='CO_CURSO', how='inner')

    # Preenche possíveis vazios gerados pelas dummies com 0
    df_consolidado = df_consolidado.fillna(0)

    return df_consolidado


def treinar_e_avaliar_modelos(df_consolidado: DataFrame):
    X = df_consolidado.drop(columns=['CO_CURSO', 'NT_GER'])
    y = df_consolidado['NT_GER']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Total de cursos válidos para análise consolidada: {len(df_consolidado)}")
    print(f"Treinamento: {len(X_train)} cursos | Teste: {len(X_test)} cursos\n")

    # --- Modelo 1: Regressão Linear Múltipla ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    print("=== MODELO 1: REGRESSÃO LINEAR MÚLTIPLA ===")
    print(f"R² (Poder explicativo): {r2_score(y_test, lr_preds):.4f}")
    print(f"RMSE (Erro médio de nota): {np.sqrt(mean_squared_error(y_test, lr_preds)):.4f}\n")

    # --- Modelo 2: Random Forest Regressor ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    print("=== MODELO 2: RANDOM FOREST REGRESSOR ===")
    print(f"R² (Poder explicativo): {r2_score(y_test, rf_preds):.4f}")
    print(f"RMSE (Erro médio de nota): {np.sqrt(mean_squared_error(y_test, rf_preds)):.4f}\n")

    importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
    print("=== TOP 15 VARIÁVEIS COM MAIOR PESO NO DESEMPENHO ===")
    print(importancias.sort_values(ascending=False).head(15))


def multi_enade_modelo_regressao(url_conexao, key_conexao):
    print("Iniciando a extração dos dados (com paginação ativada)...")
    df_arq3 = consultar_dados("tbl_arq3_2021", url_conexao, key_conexao)
    df_arq4 = consultar_dados("tbl_arq4_2021", url_conexao, key_conexao)
    df_arq21 = consultar_dados("tbl_arq21_2021", url_conexao, key_conexao)
    df_arq29 = consultar_dados("tbl_arq29_2021", url_conexao, key_conexao)

    print("Processando e agregando a matriz avançada de variáveis...")
    df_consolidado = preparar_dados_regressao(df_arq3, df_arq4, df_arq21, df_arq29)

    print("Treinando modelos com as novas dimensões...\n")
    treinar_e_avaliar_modelos(df_consolidado)


if __name__ == "__main__":
    dic_credenciais = credenciais_banco()
    multi_enade_modelo_regressao(dic_credenciais["url_banco"], dic_credenciais["key_banco"])