from mlxtend.frequent_patterns import apriori, association_rules
from util.util_db import consultar_dados, credenciais_banco
from pandas import DataFrame
import pandas as pd
import dotenv
import os
import warnings

pd.set_option('display.max_columns', None)  # Mostra todas as colunas
pd.set_option('display.max_rows', None)     # Mostra todas as linhas
pd.set_option('display.max_colwidth', None) # Impede que textos dentro da célula sejam cortados
pd.set_option('display.width', 1000)        # Estica a largura limite da tela

# Silencia os avisos de descontinuação gerados pela biblioteca do Supabase
warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")

def modelo_associacao(dataf: DataFrame, suporte_minimo: float = 0.1,confianca_minima: float = 0.5) -> pd.DataFrame:
    """
    Aplica o algoritmo Apriori para extrair regras de associação de variáveis categóricas.

    :param df: DataFrame já filtrado apenas com as colunas que o senhor deseja analisar.
    :param suporte_minimo: Frequência mínima (0 a 1) que a combinação deve ter na base.
    :param confianca_minima: Probabilidade mínima (0 a 1) de a regra ser verdadeira.
    :return: DataFrame contendo as regras
    """
    df_limpo = dataf.dropna(how="any")
    # Transforma as respostas categóricas em colunas de Verdadeiro/Falso
    df_binario = pd.get_dummies(df_limpo)
    df_binario = df_binario.astype(bool)

    # 3. Aplicamos o Apriori para encontrar os conjuntos de itens frequentes
    itemsets_frequentes = apriori(df_binario, min_support=suporte_minimo, use_colnames=True)

    # Proteção de execução caso o suporte seja muito alto para os dados atuais
    if itemsets_frequentes.empty:
        print("Nenhum padrão frequente encontrado com o suporte mínimo definido.")
        return pd.DataFrame()

    # 4. Geramos as Regras de Associação
    regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=confianca_minima)

    # Ordenamos as regras pelo 'lift' para que o senhor veja as correlações matemáticas mais fortes no topo
    regras = regras.sort_values(by="lift", ascending=False).reset_index(drop=True)
    return regras

def main(tbl_nome, url_conexao, key_conexao):
    dataf_arq4 = consultar_dados(tbl_nome ,url_conexao, key_conexao)
    list_colunas = ["QE_I57", "QE_I30", "QE_I56"]
    dataf_arq4_filtrado = dataf_arq4[list_colunas]
    regras_gerais = modelo_associacao(dataf_arq4_filtrado, 0.05, 0.5)

    regras_ordenadas = regras_gerais.sort_values(
        by=["confidence", "lift"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Exibimos o panorama geral
    print(regras_ordenadas[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

if __name__ == "__main__":
    dic_credenciais = credenciais_banco()
    main("tbl_arq4_2021", dic_credenciais["url_banco"], dic_credenciais["key_banco"])