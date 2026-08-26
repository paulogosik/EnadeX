from mlxtend.frequent_patterns import apriori, association_rules
from util.util_db import consultar_dados, credenciais_banco
from util.util_general import calcular_tempo
from pandas import DataFrame
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")

def preparar_dados_associacao(dataf: DataFrame, suporte_minimo: float = 0.1,
                              confianca_minima: float = 0.5) -> pd.DataFrame:
    try:
        list_colunas = ["QE_I57", "QE_I30", "QE_I56"]
        dataf_arq4_filtrado = dataf[list_colunas]

        df_limpo = dataf_arq4_filtrado.dropna(how="any")

        df_binario = pd.get_dummies(df_limpo)
        df_binario = df_binario.astype(bool)

        itemsets_frequentes = apriori(df_binario, min_support=suporte_minimo, use_colnames=True)

        if itemsets_frequentes.empty:
            print("Nenhum padrão frequente encontrado com o suporte mínimo definido.")
            return pd.DataFrame()
        regras = association_rules(itemsets_frequentes, metric="confidence", min_threshold=confianca_minima)
        regras = regras.sort_values(by="lift", ascending=False).reset_index(drop=True)
        print("Tratamento de dados realizado com sucesso.")
        return regras
    except Exception as e:
        print(f"Erro na funcao (preparar_dados_associacao): {e}")
        raise e

@calcular_tempo
def multi_enade_modelo_associacao(tbl_nome, url_conexao, key_conexao, suporte_minimo: float = 0.1,
                                  confianca_minima: float = 0.5, print_flag=False):
    try:
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        dataf_arq4 = consultar_dados(tbl_nome, url_conexao, key_conexao)

        regras_gerais = preparar_dados_associacao(dataf_arq4, suporte_minimo, confianca_minima)
        regras_ordenadas = regras_gerais.sort_values(
            by=["confidence", "lift"],
            ascending=[False, False]
        ).reset_index(drop=True)
        caminho_arq_csv1 = os.path.join(diretorio_atual, 'dados_associacao_binario.csv')
        regras_ordenadas.to_csv(caminho_arq_csv1, index=False)

        # Ajustes 2 e 3: Tratamento de tela para extrair o frozenset e aplicar porcentagem
        regras_formatadas = regras_ordenadas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
        # Extrai o texto limpo unindo os itens do conjunto
        regras_formatadas['antecedents'] = regras_formatadas['antecedents'].apply(lambda x: ', '.join(list(x)))
        regras_formatadas['consequents'] = regras_formatadas['consequents'].apply(lambda x: ', '.join(list(x)))
        # Formata como porcentagem e o lift com 3 casas decimais
        regras_formatadas['support'] = (regras_formatadas['support'] * 100).map("{:.2f}%".format)
        regras_formatadas['confidence'] = (regras_formatadas['confidence'] * 100).map("{:.2f}%".format)
        regras_formatadas['lift'] = regras_formatadas['lift'].map("{:.3f}".format)
        caminho_arq_csv2 = os.path.join(diretorio_atual, "dados_associacao_resultado.csv")
        regras_formatadas.to_csv(caminho_arq_csv2, index=False)

        print(regras_formatadas) if print_flag else None
    except Exception as e:
        print(f"Erro na funcao (multi_enade_modelo_associacao): {e}")


if __name__ == "__main__":
    dic_credenciais = credenciais_banco()
    multi_enade_modelo_associacao("tbl_arq4_2021", dic_credenciais["url_banco"], dic_credenciais["key_banco"],
                                  0.25, 0.7)