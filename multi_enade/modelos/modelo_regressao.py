from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from util.util_db import consultar_dados, credenciais_banco
from pandas import DataFrame
import numpy as np
import pandas as pd
import warnings

# Silencia os avisos de descontinuação gerados pela biblioteca do Supabase
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")


def preparar_dados_regressao(df_arq3: DataFrame, df_arq4: DataFrame, df_arq21: DataFrame,
                             df_arq29: DataFrame) -> DataFrame:
    """
    Trata as bases de dados realizando o mapeamento de variáveis ordinais,
    limpeza de ruídos (ausências/zeros) e agregando-as por média a nível de curso (CO_CURSO).
    """
    # 1. Agregação do Arquivo 3 (Notas de Desempenho)
    # Variável alvo: NT_GER (Nota Geral média do curso)
    df_arq3_limpo = df_arq3[['CO_CURSO', 'NT_GER']].copy()
    df_arq3_limpo['NT_GER'] = pd.to_numeric(df_arq3_limpo['NT_GER'], errors='coerce')

    # Filtro crucial: Removemos valores nulos e notas efetivamente zeradas (ausências)
    df_arq3_limpo = df_arq3_limpo[(df_arq3_limpo['NT_GER'].notna()) & (df_arq3_limpo['NT_GER'] > 0)]
    df_arq3_agg = df_arq3_limpo.groupby('CO_CURSO')['NT_GER'].mean().reset_index()

    # 2. Transformação e filtragem do Arquivo 4 (Percepção do Processo Formativo)
    # Variáveis Ordinais: QE_I63 (Infraestrutura) e QE_I57 (Domínio Docente)
    df_arq4_limpo = df_arq4[['CO_CURSO', 'QE_I63', 'QE_I57']].copy()

    # Coerção para numérico
    df_arq4_limpo['QE_I63'] = pd.to_numeric(df_arq4_limpo['QE_I63'], errors='coerce')
    df_arq4_limpo['QE_I57'] = pd.to_numeric(df_arq4_limpo['QE_I57'], errors='coerce')

    # REMOÇÃO DAS OPÇÕES QUE QUEBRAM A HIERARQUIA (6 = Não sei, 7 ou 8 = Não se aplica/anulado)
    df_arq4_limpo = df_arq4_limpo[~df_arq4_limpo['QE_I63'].isin([6, 7, 8])]
    df_arq4_limpo = df_arq4_limpo[~df_arq4_limpo['QE_I57'].isin([6, 7, 8])]

    # Calculamos a média de satisfação do curso
    df_arq4_agg = df_arq4_limpo.dropna().groupby('CO_CURSO')[['QE_I63', 'QE_I57']].mean().reset_index()

    # 3. Transformação e agregação do Arquivo 21 (Ações Afirmativas e Cotas)
    # Categórica Nominal: QE_I15 (Tipo de cota de ingresso)
    df_arq21_limpo = df_arq21[['CO_CURSO', 'QE_I15']].dropna().copy()

    # Padronização de strings para evitar duplicação de categorias
    df_arq21_limpo['QE_I15'] = df_arq21_limpo['QE_I15'].astype(str).str.strip().str.upper()

    # Geração de dummies convertidas para inteiros para a proporção funcionar limpa
    df_arq21_dummies = pd.get_dummies(df_arq21_limpo, columns=['QE_I15'], dtype=int)
    df_arq21_agg = df_arq21_dummies.groupby('CO_CURSO').mean().reset_index()

    # 4. Transformação e agregação do Arquivo 29 (Horas de Dedicação aos Estudos)
    # Variável Ordinal: QE_I23 (Tempo dedicado aos estudos além das aulas)
    df_arq29_limpo = df_arq29[['CO_CURSO', 'QE_I23']].dropna().copy()
    df_arq29_limpo['QE_I23'] = df_arq29_limpo['QE_I23'].astype(str).str.strip().str.upper()

    # Mapeamento ordinal de A a E para escala numérica de 1 a 5
    map_horas = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df_arq29_limpo['QE_I23_ordinal'] = df_arq29_limpo['QE_I23'].map(map_horas)

    # Agregação por média de horas dedicadas do curso
    df_arq29_agg = df_arq29_limpo.dropna(subset=['QE_I23_ordinal']).groupby('CO_CURSO')[
        'QE_I23_ordinal'].mean().reset_index()

    # 5. Consolidação final das tabelas utilizando a chave de agregação CO_CURSO (Inner Join)
    # Inner join garante que o modelo só veja cursos com dados completos em todas as frentes
    df_consolidado = df_arq3_agg.merge(df_arq4_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq21_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq29_agg, on='CO_CURSO', how='inner')

    return df_consolidado


def treinar_e_avaliar_modelos(df_consolidado: DataFrame):
    """
    Divide os dados, treina dois modelos de regressão (Linear clássico e RandomForest)
    e exibe as métricas de performance comparativas e importância de variáveis.
    """
    # Separação de atributos previsores (X) e rótulo alvo (y)
    X = df_consolidado.drop(columns=['CO_CURSO', 'NT_GER'])
    y = df_consolidado['NT_GER']

    # Divisão treino/teste com seed fixo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Total de cursos válidos para análise consolidada: {len(df_consolidado)}")
    print(f"Treinamento: {len(X_train)} cursos | Teste: {len(X_test)} cursos\n")

    # --- Modelo 1: Regressão Linear Múltipla ---
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    lr_r2 = r2_score(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    print("=== MODELO 1: REGRESSÃO LINEAR MÚLTIPLA ===")
    print(f"R² (Poder explicativo): {lr_r2:.4f}")
    print(f"RMSE (Erro médio de nota): {lr_rmse:.4f}\n")

    # --- Modelo 2: Random Forest Regressor ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_r2 = r2_score(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

    print("=== MODELO 2: RANDOM FOREST REGRESSOR ===")
    print(f"R² (Poder explicativo): {rf_r2:.4f}")
    print(f"RMSE (Erro médio de nota): {rf_rmse:.4f}\n")

    # Importância das variáveis no Random Forest
    importancias = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_variaveis = importancias.sort_values(ascending=False)

    print("=== PESO REAL DE CADA VARIÁVEL NO DESEMPENHO DOS CURSOS ===")
    print(top_variaveis)


def multi_enade_modelo_regressao(url_conexao, key_conexao):
    print("Iniciando a extração cuidadosa dos dados, senhor...")
    df_arq3 = consultar_dados("tbl_arq3_2021", url_conexao, key_conexao)
    df_arq4 = consultar_dados("tbl_arq4_2021", url_conexao, key_conexao)
    df_arq21 = consultar_dados("tbl_arq21_2021", url_conexao, key_conexao)
    df_arq29 = consultar_dados("tbl_arq29_2021", url_conexao, key_conexao)

    print("29", df_arq29.value_counts())
    print("21",df_arq21.value_counts())
    print("4",df_arq4.value_counts())
    print("3",df_arq3.value_counts())

    print("Preparando e agregando a base...")
    df_consolidado = preparar_dados_regressao(df_arq3, df_arq4, df_arq21, df_arq29)

    print("Iniciando o treinamento dos modelos...\n")
    treinar_e_avaliar_modelos(df_consolidado)
    return df_consolidado


if __name__ == "__main__":
    dic_credenciais = credenciais_banco()
    multi_enade_modelo_regressao(dic_credenciais["url_banco"], dic_credenciais["key_banco"])