import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from modelo_regressao import multi_enade_modelo_regressao
from util.util_db import credenciais_banco

# Configuração de estilo global
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')


def obter_dados_consolidados():
    """Busca a matriz de dados finalizada no banco."""
    dic_credenciais = credenciais_banco()
    return multi_enade_modelo_regressao(dic_credenciais["url_banco"], dic_credenciais["key_banco"])


def preparar_dados_plotagem(df, variavel_destaque='infraestrutura'):
    """
    Prepara a base para o gráfico, permitindo alternar a legenda (hue)
    entre o ARQ04 (Infraestrutura) e o ARQ21 (Cotas).
    """
    df_plot = df.copy()

    # Eixo X padrão: Dedicação
    df_plot['dedicacao'] = df_plot['QE_I23_ordinal']

    if variavel_destaque == 'infraestrutura':
        # Dados do ARQ04
        df_plot['categoria_legenda'] = np.where(
            df_plot['QE_I63'] >= 4.0,
            'Adequada/Excelente',
            'Regular/Insuficiente'
        )
        titulo_legenda = 'Infraestrutura Prática (QE_I63)'
        cores = {'Adequada/Excelente': '#023e8a', 'Regular/Insuficiente': '#f77f00'}

    elif variavel_destaque == 'cotas':
        # Dados do ARQ21 (Se + de 50% for Ampla Concorrência, classifica como Maioria Ampla)
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


def plotar_grafico(df_plot, titulo_legenda, cores, nome_arquivo='regressao_desempenho.png'):
    """Gera, exibe e salva o gráfico com base nos parâmetros escolhidos."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Plota a dispersão com a cor dinâmica (hue)
    sns.scatterplot(
        data=df_plot,
        x='dedicacao',
        y='NT_GER',
        hue='categoria_legenda',
        palette=cores,
        alpha=0.85,
        s=70,
        edgecolor='w',
        linewidth=0.5,
        ax=ax
    )

    # Plota a linha de regressão geral
    sns.regplot(
        data=df_plot,
        x='dedicacao',
        y='NT_GER',
        scatter=False,
        color='#2b2d42',
        line_kws={'linewidth': 2.5, 'linestyle': '--'},
        ax=ax
    )

    # Estética
    ax.set_title('Dedicação Individual é o Fator de Maior Impacto na Nota Geral do ENADE', fontsize=14,
                 fontweight='bold', pad=15)
    ax.set_xlabel('Média de Dedicação aos Estudos do Curso (Escala 1 a 5)', fontsize=11, fontweight='semibold',
                  labelpad=12)
    ax.set_ylabel('Nota Geral Média do Curso (NT_GER)', fontsize=11, fontweight='semibold', labelpad=12)

    # Ajuste manual da legenda para incluir a linha de regressão
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
    print(f"\n[INFO] Gráfico salvo lindamente como: {nome_arquivo}")
    plt.show()


if __name__ == "__main__":
    # 1. Carrega a base pesada
    df_base = obter_dados_consolidados()

    # 2. Senhor, basta trocar a palavra abaixo para 'infraestrutura' ou 'cotas'
    visao_escolhida = 'cotas'

    # 3. Prepara a matriz e desenha a obra de arte
    df_pronto, titulo, paleta = preparar_dados_plotagem(df_base, variavel_destaque=visao_escolhida)
    plotar_grafico(df_pronto, titulo, paleta, nome_arquivo=f'grafico_enade_{visao_escolhida}.png')