from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
import pandas as pd
import numpy as np

def show_df(dataf: DataFrame, limit_linhas=None):
    """Exibe o data frame no terminal"""
    if limit_linhas is not None:
        pd.set_option("display.max_rows", limit_linhas)
    else:
        pd.set_option("display.max_rows", 10)
    pd.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.set_option("display.width", 1000)
    print(dataf)
    print(dataf.info())


def avaliar_modelo_regressao(y_real, y_previsto, modelo=None, nomes_features=None):
    """
    Centraliza toda a análise de performance do modelo de regressão e
    exibe a importância das variáveis caso seja um modelo baseado em árvores.
    """
    # 1. Cálculo das Métricas (R², MAE e RMSE)
    r2 = r2_score(y_real, y_previsto)
    mae = mean_absolute_error(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))

    print("\n=== RESULTADOS DA AVALIAÇÃO DO MODELO ===")
    print(f"R² (Poder explicativo da variância): {r2:.4f}")
    print(f"MAE (Erro absoluto médio em notas): {mae:.4f}")
    print(f"RMSE (Erro médio penalizando grandes desvios): {rmse:.4f}")

    # 2. Análise de Importância das Variáveis (Feature Importance)
    # Verificamos se o senhor passou o modelo e se ele possui essa propriedade (como o RandomForest)
    if modelo is not None and nomes_features is not None:
        if hasattr(modelo, 'feature_importances_'):
            importancias = pd.Series(modelo.feature_importances_, index=nomes_features)
            top_variaveis = importancias.sort_values(ascending=False)

            print("\n=== PESO REAL DE CADA VARIÁVEL NO DESEMPENHO DOS CURSOS ===")
            print(top_variaveis)
    print("-" * 50)
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse}


def teste_de_parametros_regressao(X_train, y_train, X_test, y_test):
    # 1. O Menu de Ajustes (Hiperparâmetros para testar)
    parametros_grid = {
        'n_estimators': [50, 100, 200],  # Quantidade de árvores na floresta
        'max_depth': [None, 5, 10, 15],  # Profundidade (ajuda a evitar o overfitting!)
        'min_samples_split': [2, 5, 10],  # Mínimo de alunos para criar uma nova regra
        'min_samples_leaf': [1, 2, 4]  # Mínimo de alunos no resultado final da regra
    }

    # 2. Configurando o provador automático (GridSearch)
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=parametros_grid,
        scoring='neg_root_mean_squared_error',  # O objetivo dele será minimizar o RMSE
        cv=5,  # Validação cruzada em 5 partes
        n_jobs=-1  # Usa todos os núcleos do processador para ir rápido
    )

    # 3. Rodando a busca (Pode levar alguns segundos/minutos)
    print("Iniciando o Tuning do Random Forest... Aguarde!")
    grid_search.fit(X_train, y_train)

    # 4. Extraindo o modelo vencedor
    melhor_rf = grid_search.best_estimator_

    # 5. Avaliando o novo desempenho
    rf_preds_otimizado = melhor_rf.predict(X_test)
    rf_r2_otimizado = r2_score(y_test, rf_preds_otimizado)
    rf_rmse_otimizado = np.sqrt(mean_squared_error(y_test, rf_preds_otimizado))

    print("=== RESULTADO DO TUNING ===")
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print(f"Novo R² (Poder explicativo): {rf_r2_otimizado:.4f}")
    print(f"Novo RMSE (Erro médio): {rf_rmse_otimizado:.4f}")