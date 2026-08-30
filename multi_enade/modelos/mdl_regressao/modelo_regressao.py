from util.util_db import consultar_dados, credenciais_banco, upsert_supabase, truncar_tabela_supabase
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from util.util_general import calcular_tempo
from util.util_pandas import show_df
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import joblib
import shap
import os

# Silencia os avisos internos do Supabase
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")

def preparar_dados_regressao(df_arq3: DataFrame, df_arq4: DataFrame, df_arq21: DataFrame, df_arq29: DataFrame, print_flag=False) -> DataFrame:
    """
    Trata as bases de dados realizando o mapeamento de variáveis ordinais,
    limpeza de ruídos (ausências/zeros) e agregando-as por média a nível de curso (CO_CURSO).
    """
    # 1. ========== Arquivo 3 (Notas de Desempenho) ==========
    df_arq3_limpo = df_arq3[['CO_CURSO', 'NT_GER']].copy()
    df_arq3_limpo['NT_GER'] = pd.to_numeric(df_arq3_limpo['NT_GER'], errors='coerce') # Tipagem
    # Filtro: Removemos valores nulos e notas zero
    df_arq3_limpo = df_arq3_limpo[(df_arq3_limpo['NT_GER'].notna()) & (df_arq3_limpo['NT_GER'] > 0)]
    # Calculo da média da nota por curso
    df_arq3_agg = df_arq3_limpo.groupby('CO_CURSO').agg(
        NT_GER=('NT_GER', 'mean'),
        QT_ALUNOS=('NT_GER', 'count')  # Esta é a nova âncora de peso!
    ).reset_index()

    # 2. ========== Arquivo 4 (Percepção do Processo Formativo) ==========
    df_arq4_limpo = df_arq4[['CO_CURSO', 'QE_I63', 'QE_I57']].copy()
    # Tipagem
    df_arq4_limpo['QE_I63'] = pd.to_numeric(df_arq4_limpo['QE_I63'], errors='coerce')
    df_arq4_limpo['QE_I57'] = pd.to_numeric(df_arq4_limpo['QE_I57'], errors='coerce')
    # Filtro: Remoção das opções que quebram a hierarquia (6 = Não sei, 7 ou 8 = Não se aplica/anulado)
    df_arq4_limpo = df_arq4_limpo[~df_arq4_limpo['QE_I63'].isin([6, 7, 8])]
    df_arq4_limpo = df_arq4_limpo[~df_arq4_limpo['QE_I57'].isin([6, 7, 8])]
    # Calculo da média de satisfação
    df_arq4_agg = df_arq4_limpo.dropna().groupby('CO_CURSO')[['QE_I63', 'QE_I57']].mean().reset_index()

    # 3. ========== Arquivo 21 (Ações Afirmativas e Cotas) ==========
    df_arq21_limpo = df_arq21[['CO_CURSO', 'QE_I15']].dropna().copy()
    # Padronização de strings para evitar duplicação de categorias
    df_arq21_limpo['QE_I15'] = df_arq21_limpo['QE_I15'].astype(str).str.strip().str.upper()
    # Codificação da variável para números binários
    df_arq21_dummies = pd.get_dummies(df_arq21_limpo, columns=['QE_I15'], dtype=int)
    df_arq21_agg = df_arq21_dummies.groupby('CO_CURSO').mean().reset_index()

    # 4. ========== Arquivo 29 (Horas de Dedicação aos Estudos) ==========
    df_arq29_limpo = df_arq29[['CO_CURSO', 'QE_I23']].dropna().copy()
    df_arq29_limpo['QE_I23'] = df_arq29_limpo['QE_I23'].astype(str).str.strip().str.upper()
    # Mapeamento ordinal de A a E para escala numérica de 1 a 5
    map_horas = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df_arq29_limpo['QE_I23_ordinal'] = df_arq29_limpo['QE_I23'].map(map_horas)
    # Agregação por média de horas dedicadas do curso
    df_arq29_agg = df_arq29_limpo.dropna(subset=['QE_I23_ordinal']).groupby('CO_CURSO')[
        'QE_I23_ordinal'].mean().reset_index()

    # 5. InnerJoin dos Dataframes
    df_consolidado = df_arq3_agg.merge(df_arq4_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq21_agg, on='CO_CURSO', how='inner')
    df_consolidado = df_consolidado.merge(df_arq29_agg, on='CO_CURSO', how='inner')
    show_df(df_consolidado) if print_flag else None
    return df_consolidado


def treinar_e_salvar_modelo(df_consolidado: DataFrame, fatia_treino: float = 0.2,
                            caminho_modelo: str = 'rf_regressor.joblib'):
    """
    Prepara os dados, treina o Random Forest e salva (serializa) a IA no disco.
    """
    x = df_consolidado.drop(columns=['CO_CURSO', 'NT_GER', 'QT_ALUNOS'])
    y = df_consolidado['NT_GER']
    pesos = df_consolidado['QT_ALUNOS']

    X_train, X_test, y_train, y_test, pesos_train, pesos_test = train_test_split(
        x, y, pesos, test_size=fatia_treino, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    rf_model.fit(X_train, y_train, sample_weight=pesos_train)

    # Cria a pasta caso ela não exista e salva o modelo
    os.makedirs(os.path.dirname(caminho_modelo), exist_ok=True)
    joblib.dump(rf_model, caminho_modelo)
    print(f"[INFO] Inteligência Artificial treinada e salva em: {caminho_modelo}")
    return rf_model


def aplicar_modelo(df_consolidado: DataFrame, caminho_modelo: str = 'modelos/rf_regressor.joblib', print_flag=False):
    """
    Carrega o modelo do disco e aplica aos dados para gerar a nova coluna de predições.
    """
    # Carrega a IA da memória do disco
    rf_model = joblib.load(caminho_modelo)

    x = df_consolidado.drop(columns=['CO_CURSO', 'NT_GER', 'QT_ALUNOS'])

    df_pos_treino = df_consolidado.copy()
    df_pos_treino['NT_GER_PREVISTA'] = rf_model.predict(x)

    show_df(df_pos_treino) if print_flag else None
    return rf_model, df_pos_treino


def preparar_dados_plotagem(df, variavel_destaque='infraestrutura'):
    df_plot = df.copy()
    df_plot['dedicacao'] = df_plot['QE_I23_ordinal']

    if variavel_destaque == 'infraestrutura':
        df_plot['categoria_legenda'] = np.where(df_plot['QE_I63'] >= 4.0, 'Adequada/Excelente', 'Regular/Insuficiente')
        titulo_legenda = 'Infraestrutura Prática (QE_I63)'
        cores = {'Adequada/Excelente': '#023e8a', 'Regular/Insuficiente': '#f77f00'}
    elif variavel_destaque == 'cotas':
        if 'QE_I15_A' in df_plot.columns:
            df_plot['categoria_legenda'] = np.where(df_plot['QE_I15_A'] >= 0.5, 'Maioria Ampla Concorr.',
                                                    'Maioria Cotistas')
        else:
            df_plot['categoria_legenda'] = 'Dado Indisponível'
        titulo_legenda = 'Perfil de Ingresso (QE_I15)'
        cores = {'Maioria Ampla Concorr.': '#2a9d8f', 'Maioria Cotistas': '#e76f51', 'Dado Indisponível': '#cccccc'}
    else:
        raise ValueError("Senhor, o parâmetro deve ser 'infraestrutura' ou 'cotas'.")

    return df_plot, titulo_legenda, cores


def plotar_grafico_tradicional(df_plot, titulo_legenda, cores, nome_arquivo='regressao_desempenho.png'):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sns.scatterplot(data=df_plot, x='dedicacao', y='NT_GER', hue='categoria_legenda', palette=cores, alpha=0.85, s=70,
                    edgecolor='w', linewidth=0.5, ax=ax)
    sns.regplot(data=df_plot, x='dedicacao', y='NT_GER', scatter=False, color='#2b2d42',
                line_kws={'linewidth': 2.5, 'linestyle': '--'}, ax=ax)


    ax.set_xlabel('Média de Dedicação aos Estudos do Curso (QE_I23)', fontsize=11, fontweight='semibold',
                  labelpad=12)
    ax.set_ylabel('Nota Geral Média do Curso (NT_GER)', fontsize=11, fontweight='semibold', labelpad=12)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#2b2d42', linewidth=2.5, linestyle='--'))
    labels.append('Tendência Geral')

    ax.legend(handles=handles, labels=labels, title=titulo_legenda, title_fontsize='10', loc='upper left', frameon=True)
    ax.set_xlim(1, 5)
    ax.set_ylim(20, 80)
    sns.despine()
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    print(f"[INFO] Gráfico clássico salvo como: {nome_arquivo}")
    plt.show()


def plotar_grafico_shap(modelo_treinado, df_dados):
    print("[INFO] Calculando a matriz de explicabilidade SHAP... Aguarde um instante.")
    colunas_ignoradas = ['CO_CURSO', 'NT_GER', 'QT_ALUNOS', 'NT_GER_PREVISTA']
    X = df_dados.drop(columns=[col for col in colunas_ignoradas if col in df_dados.columns])

    explainer = shap.TreeExplainer(modelo_treinado)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(10, 6.5))
    plt.title('Peso de Cada Fator na Nota do ENADE (NT_GER)', fontsize=14, fontweight='bold', pad=20)
    shap.summary_plot(shap_values, X, show=False)

    plt.tight_layout()
    #plt.savefig('../grafico_shap_explicabilidade.png', dpi=150, bbox_inches='tight')
    print("[INFO] Gráfico SHAP salvo lindamente como: grafico_shap_explicabilidade.png")
    plt.show()

@calcular_tempo
def multi_enade_modelo_regressao(url_conexao, key_conexao, flag_exe_treino=False):
    print("Extraindo dados do Supabase...")
    df_arq3 = consultar_dados("tbl_arq3_2021", url_conexao, key_conexao)
    df_arq4 = consultar_dados("tbl_arq4_2021", url_conexao, key_conexao)
    df_arq21 = consultar_dados("tbl_arq21_2021", url_conexao, key_conexao)
    df_arq29 = consultar_dados("tbl_arq29_2021", url_conexao, key_conexao)

    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_do_modelo = os.path.join(diretorio_atual, 'rf_regressor.joblib')

    print("Preparando e agregando a base...")
    df_preparado = preparar_dados_regressao(df_arq3, df_arq4, df_arq21, df_arq29)
    df_preparado.to_csv("dados_regressao_refinados.csv", index=False)

    treinar_e_salvar_modelo(df_preparado, 0.2, caminho_do_modelo) if flag_exe_treino else None

    print("Iniciando o treinamento dos modelos...\n")
    modelo_treinado, df_pos_treino = aplicar_modelo(df_preparado, caminho_modelo=caminho_do_modelo)
    df_pos_treino.to_csv("dados_regressao_pos_treino.csv", index=False)

    upsert_supabase(df_pos_treino, "tbl_multi_enade_regressao", url_conexao, key_conexao)

    visao = 'cotas'
    df_pronto, titulo, paleta = preparar_dados_plotagem(df_pos_treino, variavel_destaque=visao)
    plotar_grafico_tradicional(df_pronto, titulo, paleta, nome_arquivo=f'grafico_enade_{visao}.png')
    # 2. Renderiza o Gráfico de Explicabilidade (SHAP)
    plotar_grafico_shap(modelo_treinado, df_pos_treino)

    print("\nPipeline de Regressão finalizado com absoluto sucesso, senhor Dan!")


if __name__ == "__main__":
    dic_credenciais = credenciais_banco()
    multi_enade_modelo_regressao(dic_credenciais["url_banco"], dic_credenciais["key_banco"])