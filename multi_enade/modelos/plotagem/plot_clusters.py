from multi_enade.modelos.modelo_clusters import multi_enade_modelo_clusters
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configuração visual elegante para os gráficos
sns.set_theme(style="whitegrid", palette="muted")


def gerar_grafico_clusters(df_dados: pd.DataFrame, num_clusters: int = 3):
    """
    Padroniza os dados, aplica o algoritmo K-Means e plota os cursos em um gráfico 2D usando PCA.

    :param df_dados: O DataFrame gerado pela função 'preparar_dados_cluster_triplo'.
    :param num_clusters: A quantidade de grupos que o senhor deseja que a IA identifique.
    """
    print(f"🎨 Iniciando a clusterização para {num_clusters} grupos ocultos...")

    # 1. Isolamos o CO_CURSO, pois o algoritmo matemático só aceita números
    df_matematica = df_dados.drop(columns=['CO_CURSO'])

    # 2. Padronização (StandardScaler):
    # Coloca notas (0-100) e porcentagens (0-100) na mesma escala de peso estatístico
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_matematica)

    # 3. Treinamento do K-Means para encontrar os padrões pedagógicos
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters_encontrados = kmeans.fit_predict(dados_padronizados)

    print("\n✨ Calculando o perfil exato de cada grupo pedagógico...")

    # 1. Criamos uma cópia segura dos dados originais e adicionamos a coluna com o número do cluster
    df_raio_x = df_dados.copy()
    df_raio_x['Cluster_ID'] = clusters_encontrados

    # 2. Removemos o CO_CURSO (pois não calculamos média de texto) e agrupamos pela nova coluna
    perfil_clusters = df_raio_x.drop(columns=['CO_CURSO']).groupby('Cluster_ID').mean()

    # 3. Transpomos a tabela (.T) para que os Clusters fiquem nas colunas e as variáveis nas linhas.
    # Isso torna a leitura no terminal infinitamente mais fácil para o senhor!
    perfil_formatado = perfil_clusters.T

    # 4. Arredondamos para 2 casas decimais para manter a tela limpa e elegante
    print("\n📊 PERFIL NUMÉRICO DOS CLUSTERS:")
    print(perfil_formatado.round(2))
    print("=" * 60)

    # 4. Redução de Dimensionalidade (PCA):
    # Esmaga as dezenas de colunas em apenas 2 para conseguirmos desenhar na tela
    pca = PCA(n_components=2)
    dados_2d = pca.fit_transform(dados_padronizados)

    # 5. Montamos um DataFrame temporário apenas para o gráfico
    df_plot = pd.DataFrame({
        'Eixo X (Variação Principal)': dados_2d[:, 0],
        'Eixo Y (Variação Secundária)': dados_2d[:, 1],
        'Perfil Identificado': [f'Cluster {c}' for c in clusters_encontrados]
    })

    print("📈 Renderizando o gráfico de dispersão...")

    # 6. Criação do gráfico com Seaborn
    plt.figure(figsize=(10, 7))
    grafico = sns.scatterplot(
        data=df_plot,
        x='Eixo X (Variação Principal)',
        y='Eixo Y (Variação Secundária)',
        hue='Perfil Identificado',
        palette='viridis',
        s=100,  # Tamanho das bolinhas
        alpha=0.8,  # Transparência para ver sobreposições
        edgecolor='black'
    )

    # Detalhes de acabamento e títulos
    plt.title('Mapeamento de Cursos: Desempenho, Infraestrutura e Diversidade', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(f"Eixo X (Explica {pca.explained_variance_ratio_[0] * 100:.1f}% dos dados)", fontsize=11)
    plt.ylabel(f"Eixo Y (Explica {pca.explained_variance_ratio_[1] * 100:.1f}% dos dados)", fontsize=11)
    plt.legend(title='Grupos Pedagógicos', title_fontsize='12', fontsize='10')
    plt.tight_layout()

    # Exibe a janela gráfica para o senhor
    plt.show()


# Exemplo de como o senhor conectará este arquivo ao anterior
if __name__ == "__main__":
    # Aqui o senhor importaria a sua função de preparação do outro arquivo
    df_preparado = multi_enade_modelo_clusters()
    gerar_grafico_clusters(df_preparado, num_clusters=4)
    print("Módulo de plotagem visual pronto para ser importado e executado!")