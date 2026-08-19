import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
import shap

# Configuração de estilo global
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')


# ==============================================================================
# JEITO 1: O GRÁFICO CLÁSSICO DE NEGÓCIOS (Preservando sua obra de arte)
# ==============================================================================

def preparar_dados_plotagem(df, variavel_destaque='infraestrutura'):
    """
    Prepara a base para o gráfico clássico, permitindo alternar a legenda (hue).
    """
    df_plot = df.copy()

    # Eixo X padrão: Dedicação
    df_plot['dedicacao'] = df_plot['QE_I23_ordinal']

    if variavel_destaque == 'infraestrutura':
        df_plot['categoria_legenda'] = np.where(
            df_plot['QE_I63'] >= 4.0,
            'Adequada/Excelente',
            'Regular/Insuficiente'
        )
        titulo_legenda = 'Infraestrutura Prática (QE_I63)'
        cores = {'Adequada/Excelente': '#023e8a', 'Regular/Insuficiente': '#f77f00'}

    elif variavel_destaque == 'cotas':
        if 'QE_I15_A' in df_plot.columns:
            df_plot['categoria_legenda'] = np.where(
                df_plot['QE_I15_A'] >= 0.5,
                'Maioria Ampla Concorr.',
                'Maioria Cotistas'
            )
        else:
            df_plot['categoria_legenda'] = 'Dado Indisponível'

        titulo_legenda = 'Perfil de Ingresso (QE_I15)'
        cores = {'Maioria Ampla Concorr.': '#2a9d8f', 'Maioria Cotistas': '#e76f51', 'Dado Indisponível': '#cccccc'}

    else:
        raise ValueError("Senhor, o parâmetro deve ser 'infraestrutura' ou 'cotas'.")

    return df_plot, titulo_legenda, cores


def plotar_grafico_tradicional(df_plot, titulo_legenda, cores, nome_arquivo='regressao_desempenho.png'):
    """Gera, exibe e salva o gráfico de dispersão com base nos parâmetros escolhidos."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    sns.scatterplot(
        data=df_plot, x='dedicacao', y='NT_GER', hue='categoria_legenda',
        palette=cores, alpha=0.85, s=70, edgecolor='w', linewidth=0.5, ax=ax
    )

    sns.regplot(
        data=df_plot, x='dedicacao', y='NT_GER', scatter=False,
        color='#2b2d42', line_kws={'linewidth': 2.5, 'linestyle': '--'}, ax=ax
    )

    ax.set_title('Dedicação Individual é o Fator de Maior Impacto na Nota Geral', fontsize=14, fontweight='bold',
                 pad=15)
    ax.set_xlabel('Média de Dedicação aos Estudos do Curso (Escala 1 a 5)', fontsize=11, fontweight='semibold',
                  labelpad=12)
    ax.set_ylabel('Nota Geral Média do Curso (NT_GER)', fontsize=11, fontweight='semibold', labelpad=12)

    handles, labels = ax.get_legend_handles_labels()
    trend_line_handle = Line2D([0], [0], color='#2b2d42', linewidth=2.5, linestyle='--')
    handles.append(trend_line_handle)
    labels.append('Tendência Geral')

    ax.legend(handles=handles, labels=labels, title=titulo_legenda, title_fontsize='10', loc='upper left', frameon=True)
    ax.set_xlim(1, 5)
    ax.set_ylim(20, 80)
    sns.despine()
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Gráfico clássico salvo lindamente como: {nome_arquivo}")
    plt.show()


# ==============================================================================
# JEITO 2: O GRÁFICO DA INTELIGÊNCIA ARTIFICIAL (Explicabilidade SHAP)
# ==============================================================================

def plotar_grafico_shap(modelo_treinado, df_dados):
    """
    Lê a mente do Random Forest e desenha exatamente o peso de cada variável.
    """
    print("\n[INFO] Calculando a matriz de explicabilidade SHAP... Isso pode levar alguns segundos.")

    # 1. Isolamos apenas as variáveis que o modelo usou para aprender
    # Ignoramos notas, códigos e quantidade de alunos
    colunas_ignoradas = ['CO_CURSO', 'NT_GER', 'QT_ALUNOS', 'NT_GER_PREVISTA']
    X = df_dados.drop(columns=[col for col in colunas_ignoradas if col in df_dados.columns])

    # 2. Instanciamos o explicador do SHAP focado em modelos de árvore
    explainer = shap.TreeExplainer(modelo_treinado)
    shap_values = explainer.shap_values(X)

    # 3. Renderizamos a obra de arte analítica
    plt.figure(figsize=(10, 6.5))
    plt.title('O Que a IA Aprendeu: Peso de Cada Fator na Nota do ENADE (SHAP)', fontsize=14, fontweight='bold', pad=20)

    # O summary_plot desenha o gráfico de impacto
    shap.summary_plot(shap_values, X, show=False)

    plt.tight_layout()
    plt.savefig('grafico_shap_explicabilidade.png', dpi=150, bbox_inches='tight')
    print("[INFO] Gráfico SHAP salvo lindamente como: grafico_shap_explicabilidade.png")
    plt.show()


# ==============================================================================
# EXECUÇÃO DO MÓDULO VISUAL
# ==============================================================================
if __name__ == "__main__":
    # Observação: Para rodar este arquivo sozinho, o senhor precisaria importar
    # a sua função 'multi_enade_modelo_regressao' e pegar o modelo treinado.
    # Exemplo simulado de como o fluxo funcionaria:

    from multi_enade.modelos.modelo_regressao import multi_enade_modelo_regressao
    from util.util_db import credenciais_banco

    dic_credenciais = credenciais_banco()
    modelo, df_treinado = multi_enade_modelo_regressao(dic_credenciais["url_banco"], dic_credenciais["key_banco"])

    # 1. Plot Clássico
    visao_escolhida = 'cotas'
    df_pronto, titulo, paleta = preparar_dados_plotagem(df_treinado, variavel_destaque=visao_escolhida)
    plotar_grafico_tradicional(df_pronto, titulo, paleta, nome_arquivo=f'grafico_enade_{visao_escolhida}.png')

    # 2. Plot SHAP
    plotar_grafico_shap(modelo, df_treinado)