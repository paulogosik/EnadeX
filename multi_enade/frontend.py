import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================================================================
# CONFIGURAÇÃO GERAL E ESTÉTICA
# ==============================================================================
st.set_page_config(
    page_title="MultiENADE - Analytics",
    page_icon="🎓",
    layout="wide"
)

# CSS Poderoso: Fundo branco E letras escuras (para anular o modo escuro)
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #2b2d42 !important;
    }
    /* Tratamento especial para os botões das abas */
    .stTabs [data-baseweb="tab"] p {
        color: #6c757d !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] p {
        color: #2b2d42 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Estética global dos gráficos (sua versão clássica)
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')

# Cabeçalho Principal
st.title("🎓 Projeto MultiENADE")
st.markdown("### Mapeamento Multidimensional e Descoberta de Padrões")
st.divider()

# Abas de Navegação
aba_regressao, aba_clusters, aba_associacao = st.tabs([
    "📈 Impacto no Desempenho (Regressão)",
    "🧩 Perfis Latentes (Clusters)",
    "🔗 Regras de Associação"
])

# ==============================================================================
# MÓDULO 1: REGRESSÃO
# ==============================================================================
with aba_regressao:
    st.header("Análise Preditiva da Nota Geral")

    visao_escolhida = st.radio(
        "Selecione a variável de destaque para colorir o gráfico:",
        options=["cotas", "infraestrutura"],
        format_func=lambda x: "Perfil de Ingresso (Cotas)" if x == "cotas" else "Infraestrutura Prática",
        horizontal=True
    )

    try:
        df = pd.read_csv("modelos/dados/dados_regressao_pos_treino.csv")
        df_plot = df.copy()
        df_plot['dedicacao'] = df_plot['QE_I23_ordinal']

        if visao_escolhida == 'infraestrutura':
            df_plot['categoria_legenda'] = np.where(df_plot['QE_I63'] >= 4.0, 'Adequada/Excelente',
                                                    'Regular/Insuficiente')
            titulo_legenda = 'Infraestrutura Prática (QE_I63)'
            cores = {'Adequada/Excelente': '#023e8a', 'Regular/Insuficiente': '#f77f00'}
        else:
            if 'QE_I15_A' in df_plot.columns:
                df_plot['categoria_legenda'] = np.where(df_plot['QE_I15_A'] >= 0.5, 'Maioria Ampla Concorr.',
                                                        'Maioria Cotistas')
            else:
                df_plot['categoria_legenda'] = 'Dado Indisponível'
            titulo_legenda = 'Perfil de Ingresso (QE_I15)'
            cores = {'Maioria Ampla Concorr.': '#2a9d8f', 'Maioria Cotistas': '#e76f51', 'Dado Indisponível': '#cccccc'}

        # Gráfico esticado horizontalmente e achatado verticalmente para não dar scroll
        fig, ax = plt.subplots(figsize=(12, 4.5))

        sns.scatterplot(data=df_plot, x='dedicacao', y='NT_GER', hue='categoria_legenda', palette=cores, alpha=0.85,
                        s=70, edgecolor='w', linewidth=0.5, ax=ax)
        sns.regplot(data=df_plot, x='dedicacao', y='NT_GER', scatter=False, color='#2b2d42',
                    line_kws={'linewidth': 2.5, 'linestyle': '--'}, ax=ax)

        ax.set_xlabel('Média de Dedicação aos Estudos do Curso (Escala 1 a 5)', fontsize=11, fontweight='semibold',
                      labelpad=12)
        ax.set_ylabel('Nota Geral Média do Curso (NT_GER)', fontsize=11, fontweight='semibold', labelpad=12)

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='#2b2d42', linewidth=2.5, linestyle='--'))
        labels.append('Tendência Geral')

        ax.legend(handles=handles, labels=labels, title=titulo_legenda, title_fontsize='10', loc='upper left',
                  frameon=True)
        ax.set_xlim(1, 5)
        ax.set_ylim(20, 80)
        sns.despine()

        st.pyplot(fig)

        # Legenda posicionada logo abaixo
        st.info("""
        **📚 Dicionário de Variáveis:**
        * **`NT_GER`**: Nota Geral Média do Curso | **`QE_I23`**: Horas de Dedicação aos Estudos (Eixo X) | **`QE_I15`**: Perfil de Ingresso (Cotas vs. Ampla Concorrência) | **`QE_I63`**: Avaliação da Infraestrutura Prática
        """)

    except FileNotFoundError:
        st.error("⚠️ Senhor, não encontrei o arquivo `dados_regressao_pos_treino.csv`.")

# ==============================================================================
# MÓDULO 2: CLUSTERS
# ==============================================================================
with aba_clusters:
    st.header("Exploração de Perfis Pedagógicos")
    st.markdown("#### Mapeamento Espacial de Cursos (PCA)")

    try:
        df_clusters = pd.read_csv("modelos/mdl_clusters/dados_clusters_treinado.csv")

        features = df_clusters.drop(columns=['CO_CURSO', 'Cluster_ID'], errors='ignore')
        clusters_id = df_clusters['Cluster_ID']

        scaler = StandardScaler()
        dados_padronizados = scaler.fit_transform(features)

        pca = PCA(n_components=2)
        dados_2d = pca.fit_transform(dados_padronizados)

        df_plot_cluster = pd.DataFrame({
            'Eixo X': dados_2d[:, 0],
            'Eixo Y': dados_2d[:, 1],
            'Perfil Identificado': [f'Cluster {c}' for c in clusters_id]
        })

        # Gráfico esticado horizontalmente e achatado verticalmente para não dar scroll
        fig_cluster, ax_cluster = plt.subplots(figsize=(12, 4.5))

        sns.scatterplot(
            data=df_plot_cluster, x='Eixo X', y='Eixo Y', hue='Perfil Identificado',
            palette='viridis', s=100, alpha=0.8, edgecolor='black', ax=ax_cluster
        )
        ax_cluster.set_xlabel(f"Eixo X (Explica {pca.explained_variance_ratio_[0] * 100:.1f}%)", fontsize=11)
        ax_cluster.set_ylabel(f"Eixo Y (Explica {pca.explained_variance_ratio_[1] * 100:.1f}%)", fontsize=11)
        ax_cluster.legend(title='Grupos Pedagógicos', title_fontsize='10', frameon=True)
        sns.despine()

        st.pyplot(fig_cluster)

        # Legenda posicionada logo abaixo
        st.info("""
        **📚 Dicionário de Variáveis:**
        * **`NT_GER`**: Nota Geral Média do Curso | **`Eixos PCA`**: variveis de comportamento de Infraestrutura, Cotas e Desempenho.
        """)

    except FileNotFoundError:
        st.error("⚠️ não encontrei o arquivo `dados_clusters_treinado.csv`.")

# ==============================================================================
# MÓDULO 3: REGRAS DE ASSOCIAÇÃO
# ==============================================================================
with aba_associacao:
    st.header("Padrões de Associação (Algoritmo Apriori)")
    st.markdown("Descoberta de regras de coocorrência entre variáveis com suporte e confiança mínima.")

    try:
        df_regras = pd.read_csv("modelos/mdl_associacao/dados_associacao_resultado.csv")

        # Altura reduzida (300) para caber na tela com o dicionário sem rolagem geral
        st.dataframe(df_regras, use_container_width=True, height=300)

        # Legenda posicionada logo abaixo
        st.info("""
        **📚 Dicionário de Variáveis:**
        * **`QE_I57`**: Domínio docente do conteúdo | **`QE_I56`**: Disponibilidade dos docentes | **`QE_I30`**: Experiência acadêmica inovadoras.
        """)

    except FileNotFoundError:
        st.error("⚠️ Senhor, não encontrei o arquivo `dados_associacao_resultado.csv`.")