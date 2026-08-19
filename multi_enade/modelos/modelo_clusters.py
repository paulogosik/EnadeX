from util.util_db import consultar_dados, credenciais_banco
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from util.util_pandas import show_df
from sklearn.cluster import KMeans
from pandas import DataFrame
import seaborn as sns
import pandas as pd
import warnings

# Silencia os avisos internos do Supabase
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")

def preparar_dados_cluster_triplo(df_arq3: DataFrame, df_arq4: DataFrame, df_arq21: DataFrame, print_flag=False) -> DataFrame:
    """
    Consolida e transforma os dados de Desempenho (Arq 3), Infraestrutura (Arq 4)
    e Diversidade (Arq 21) por Curso para aplicação de Clusterização.
    """
    # 1. Tratamento Arq 3: Isola a nota geral, e calcula a média
    df_3 = df_arq3[['CO_CURSO', 'NT_GER']].dropna()
    df_3_agrupado = df_3.groupby('CO_CURSO')['NT_GER'].mean().reset_index()

    # 2. Tratamento Arq 4: Binariza a variável categórica e calcula a porcentagem
    df_4 = df_arq4[['CO_CURSO', 'QE_I63']].dropna()
    df_binario_4 = pd.get_dummies(df_4, columns=['QE_I63'])
    colunas_dummies_4 = [col for col in df_binario_4.columns if col != 'CO_CURSO'] # Lista as colunas criadas no (get_dummies)
    df_binario_4[colunas_dummies_4] = df_binario_4[colunas_dummies_4].astype(float) # Tipagem para float
    df_4_agrupado = df_binario_4.groupby('CO_CURSO')[colunas_dummies_4].mean().reset_index()

    # 3. Tratamento Arq 21: Binariza a variável categórica e calcula a porcentagem
    df_21 = df_arq21[['CO_CURSO', 'QE_I15']].dropna()
    df_binario_21 = pd.get_dummies(df_21, columns=['QE_I15'])
    colunas_dummies_21 = [col for col in df_binario_21.columns if col != 'CO_CURSO'] # Lista as colunas criadas no (get_dummies)
    df_binario_21[colunas_dummies_21] = df_binario_21[colunas_dummies_21].astype(float) # Tipagem para float
    df_21_agrupado = df_binario_21.groupby('CO_CURSO')[colunas_dummies_21].mean().reset_index()
    # 4. InnerJoin dos Dataframes
    df_merge_1 = pd.merge(df_3_agrupado, df_4_agrupado, on='CO_CURSO', how='inner')
    df_cluster_final = pd.merge(df_merge_1, df_21_agrupado, on='CO_CURSO', how='inner')

    show_df(df_cluster_final) if print_flag else None
    return df_cluster_final


def treinar_clusters(df_dados: pd.DataFrame, num_clusters: int = 3, print_flag=False) -> pd.DataFrame:
    """
    Padroniza os dados, aplica o algoritmo K-Means e calcula o perfil de cada grupo.
    Retorna o DataFrame enriquecido com a classificação (Cluster_ID).
    """
    print(f"Iniciando o treinamento do K-Means para {num_clusters} grupos...")

    df_matematica = df_dados.drop(columns=['CO_CURSO'])
    # 2. Padronização (StandardScaler)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_matematica)
    # 3. Treinamento do K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters_encontrados = kmeans.fit_predict(dados_padronizados)

    # 4. Criamos uma cópia segura e adicionamos a nova coluna com os agrupamentos
    df_clusterizado = df_dados.copy()
    df_clusterizado['Cluster_ID'] = clusters_encontrados

    print("\nCalculando o perfil exato de cada grupo pedagógico...")
    perfil_clusters = df_clusterizado.drop(columns=['CO_CURSO']).groupby('Cluster_ID').mean()

    print("\nPERFIL NUMÉRICO DOS CLUSTERS:")
    print(perfil_clusters.T.round(2))
    print("=" * 60)
    show_df(df_clusterizado) if print_flag else None
    return df_clusterizado

def plotar_grafico_clusters(df_clusterizado: pd.DataFrame):
    """
    Recebe o DataFrame com os clusters definidos, aplica PCA para redução
    de dimensionalidade e renderiza o gráfico de dispersão 2D.
    """
    print("Preparando dados para visualização...")

    # 1. Separamos as variáveis matemáticas (ignorando CO_CURSO e o Cluster_ID)
    features = df_clusterizado.drop(columns=['CO_CURSO', 'Cluster_ID'])
    clusters = df_clusterizado['Cluster_ID']

    # 2. Padronizamos as variáveis para garantir que o PCA desenhe o eixo perfeitamente
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(features)

    # 3. Redução de Dimensionalidade (PCA) para apenas 2 componentes (X e Y)
    pca = PCA(n_components=2)
    dados_2d = pca.fit_transform(dados_padronizados)

    # 4. Montamos um DataFrame temporário exclusivamente para o gráfico
    df_plot = pd.DataFrame({
        'Eixo X (Variação Principal)': dados_2d[:, 0],
        'Eixo Y (Variação Secundária)': dados_2d[:, 1],
        'Perfil Identificado': [f'Cluster {c}' for c in clusters]
    })
    # 5. Configuração e criação do gráfico
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_plot,
        x='Eixo X (Variação Principal)',
        y='Eixo Y (Variação Secundária)',
        hue='Perfil Identificado',
        palette='viridis',
        s=100,
        alpha=0.8,
        edgecolor='black'
    )
    # 6. Detalhes de acabamento
    plt.title('Mapeamento de Cursos: Desempenho, Infraestrutura e Diversidade', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"Eixo X (Explica {pca.explained_variance_ratio_[0] * 100:.1f}% dos dados)", fontsize=11)
    plt.ylabel(f"Eixo Y (Explica {pca.explained_variance_ratio_[1] * 100:.1f}% dos dados)", fontsize=11)
    plt.legend(title='Grupos Pedagógicos', title_fontsize='12', fontsize='10')
    plt.tight_layout()
    plt.show()


def multi_enade_modelo_clusters() -> DataFrame:
    dic_credenciais = credenciais_banco()
    url = dic_credenciais["url_banco"]
    key = dic_credenciais["key_banco"]
    # 1. Consulta dos dados
    print("Extraindo dados do Supabase...")
    dataf_arq3 = consultar_dados("tbl_arq3_2021", url, key)
    dataf_arq4 = consultar_dados("tbl_arq4_2021", url, key)
    dataf_arq21 = consultar_dados("tbl_arq21_2021", url, key)
    # 2. Preparação dos dados
    print("Preparando os dados...")
    df_pronto_para_cluster = preparar_dados_cluster_triplo(dataf_arq3, dataf_arq4, dataf_arq21)
    df_pronto_para_cluster.to_csv("dados/dados_clusters.csv", index=False)
    # 3. Treinamento dos clusters
    df_clusterizado = treinar_clusters(df_pronto_para_cluster, 3)
    df_clusterizado.to_csv("dados/dados_clusters_treinado.csv", index=False)
    # 4. Plotagem
    plotar_grafico_clusters(df_clusterizado)



if __name__ == "__main__":
    multi_enade_modelo_clusters()