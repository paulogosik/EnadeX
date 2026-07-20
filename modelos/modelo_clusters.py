import pandas as pd
import warnings
from pandas import DataFrame
from util.util_db import consultar_dados, credenciais_banco

# Configurações de exibição do Pandas para controle total no terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

# Silencia os avisos internos do Supabase
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")


def preparar_dados_cluster_triplo(
        df_arq3: DataFrame,
        df_arq4: DataFrame,
        df_arq21: DataFrame
) -> DataFrame:
    """
    Consolida e transforma os dados de Desempenho (Arq 3), Infraestrutura (Arq 4)
    e Diversidade (Arq 21) por Curso para aplicação de Clusterização.
    """

    print("🚀 Processando Arquivo 3 (Desempenho: NT_GER)...")
    # 1. Tratamento Arq 3: Isola a nota geral, remove nulos e calcula a média
    df_3 = df_arq3[['CO_CURSO', 'NT_GER']].dropna()
    df_3['CO_CURSO'] = df_3['CO_CURSO'].astype(str)
    df_3['NT_GER'] = df_3['NT_GER'].astype(float)
    df_3_agrupado = df_3.groupby('CO_CURSO')['NT_GER'].mean().reset_index()

    print("🚀 Processando Arquivo 4 (Infraestrutura: QE_I63)...")
    # 2. Tratamento Arq 4: Binariza a variável categórica e calcula a porcentagem
    df_4 = df_arq4[['CO_CURSO', 'QE_I63']].dropna()
    df_4['CO_CURSO'] = df_4['CO_CURSO'].astype(str)

    df_binario_4 = pd.get_dummies(df_4, columns=['QE_I63'])
    colunas_dummies_4 = [col for col in df_binario_4.columns if col != 'CO_CURSO']

    df_binario_4[colunas_dummies_4] = df_binario_4[colunas_dummies_4].astype(float)
    df_4_agrupado = df_binario_4.groupby('CO_CURSO')[colunas_dummies_4].mean().reset_index()
    df_4_agrupado[colunas_dummies_4] = df_4_agrupado[colunas_dummies_4] * 100

    print("🚀 Processando Arquivo 21 (Diversidade: QE_I15)...")
    # 3. Tratamento Arq 21: Binariza a variável categórica e calcula a porcentagem
    df_21 = df_arq21[['CO_CURSO', 'QE_I15']].dropna()
    df_21['CO_CURSO'] = df_21['CO_CURSO'].astype(str)

    df_binario_21 = pd.get_dummies(df_21, columns=['QE_I15'])
    colunas_dummies_21 = [col for col in df_binario_21.columns if col != 'CO_CURSO']

    df_binario_21[colunas_dummies_21] = df_binario_21[colunas_dummies_21].astype(float)
    df_21_agrupado = df_binario_21.groupby('CO_CURSO')[colunas_dummies_21].mean().reset_index()
    df_21_agrupado[colunas_dummies_21] = df_21_agrupado[colunas_dummies_21] * 100

    print("🔗 Realizando o Inner Join triplo das bases consolidadas...")
    # 4. Cruzamento exato usando CO_CURSO como elo de ligação
    df_merge_1 = pd.merge(df_3_agrupado, df_4_agrupado, on='CO_CURSO', how='inner')
    df_cluster_final = pd.merge(df_merge_1, df_21_agrupado, on='CO_CURSO', how='inner')

    return df_cluster_final


def modelo_clusters():
    dic_credenciais = credenciais_banco()
    url = dic_credenciais["url_banco"]
    key = dic_credenciais["key_banco"]

    print("📥 Extraindo dados do Supabase...")
    # Buscando as tabelas originais - Substitua pelos nomes exatos das suas tabelas
    dataf_arq3 = consultar_dados("tbl_arq3_2021", url, key)
    dataf_arq4 = consultar_dados("tbl_arq4_2021", url, key)
    dataf_arq21 = consultar_dados("tbl_arq21_2021", url, key)

    # Aciona a função que fará a mágica da engenharia de recursos
    df_pronto_para_cluster = preparar_dados_cluster_triplo(dataf_arq3, dataf_arq4, dataf_arq21)

    print("\n✨ Base unificada e perfeitamente estruturada para Clusterização!")
    print(f"Total de Cursos consolidados: {len(df_pronto_para_cluster)}")
    print("\n👀 Visão geral das variáveis prontas para o K-Means (ou similar):")
    print(df_pronto_para_cluster.head())

    return df_pronto_para_cluster


if __name__ == "__main__":
    main()