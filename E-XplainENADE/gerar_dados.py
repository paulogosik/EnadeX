"""
Gera a base de dados embutida do recorte do TCC — nível de CURSO.

Recorte fixo (E-XplainENADE):
  - ENADE 2021
  - Cursos: Ciência da Computação (4004) + Sistemas de Informação (4006)  [CO_GRUPO]
  - Brasil inteiro — sem filtro de região (decisão registrada no
    DEVELOPMENT.md em 2026-08-27; o mockup original do sistema nunca previu
    um filtro de região para o E-XplainENADE, ao contrário do MultiENADE)

Cada um dos 13 arquivos brutos do INEP é agregado por CO_CURSO (a única
chave de junção válida entre arquivos — ver modules/etl.py e o
DEVELOPMENT.md, descoberta de 2026-08-20) e os agregados são unidos por
essa chave. Toda a recodificação (letra→número, binarizações, dummies de
turno) é aplicada por modules.loader.preprocess() antes da agregação —
o resultado já vem pronto para modelagem, com nomes amigáveis.

Saída: dados/enade_2021_ccsi_cursos.csv.gz — uma linha por curso
(~700 cursos, ~1 MB — versionado no git).

Uso:
    python gerar_dados.py
    python gerar_dados.py --fonte "caminho/para/2.DADOS"
    python gerar_dados.py --saida dados/meu_arquivo.csv.gz
"""
import argparse
import sys
import time
from pathlib import Path

from modules import etl

_ROOT = Path(__file__).parent
# Dados brutos/gerados não são essenciais à execução do sistema (que usa o
# Supabase por padrão) e por isso vivem fora do código, em docs/E-XplainENADE/
# (2026-08-30 — ver DEVELOPMENT.md).
_DOCS_DIR = _ROOT.parent / "docs" / "E-XplainENADE"

GRUPOS = [4004, 4006]  # CC + SI
SAIDA_PADRAO = _DOCS_DIR / "dados" / "enade_2021_ccsi_cursos.csv.gz"

# Locais onde os microdados brutos costumam estar neste projeto
_FONTES_PADRAO = [
    _DOCS_DIR / "microdados_Enade_2021_LGPD" / "2.DADOS",
    _DOCS_DIR / "gerar_csv" / "microdados_Enade_2021_LGPD" / "2.DADOS",
]


def _localizar_fonte(arg: str) -> Path:
    if arg:
        p = Path(arg)
        if p.exists():
            return p
        sys.exit(f"Erro: pasta de origem não encontrada: {p}")
    for p in _FONTES_PADRAO:
        if p.exists():
            return p
    sys.exit(
        "Erro: microdados brutos não encontrados.\n"
        "Esperado em uma destas pastas:\n"
        + "\n".join(f"  - {p}" for p in _FONTES_PADRAO)
        + "\nOu informe o caminho com: python gerar_dados.py --fonte <pasta 2.DADOS>"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gera a base embutida do recorte do TCC (2021 · CC+SI · Brasil inteiro · por curso)"
    )
    parser.add_argument("--fonte", "-f", default=None,
                        help="Pasta com os arquivos microdados2021_arq*.txt")
    parser.add_argument("--saida", "-o", default=None,
                        help=f"Arquivo de saída (padrão: {SAIDA_PADRAO.relative_to(_ROOT)})")
    args = parser.parse_args()

    fonte = _localizar_fonte(args.fonte)
    saida = Path(args.saida) if args.saida else SAIDA_PADRAO

    print("=" * 64)
    print("  E-XplainENADE — Geração da base do recorte (ENADE 2021)")
    print("=" * 64)
    print(f"  Fonte   : {fonte}")
    print(f"  Saída   : {saida}")
    print(f"  Recorte : grupos {GRUPOS} (CC+SI) · Brasil inteiro · agregado por CO_CURSO")
    print()

    t0 = time.time()
    print("Lendo os 13 arquivos e agregando por curso (join por CO_CURSO, não por posição)...")
    df = etl.load_raw(grupos=GRUPOS, raw_dir=fonte)
    print(f"  Agregação concluída: {len(df):,} cursos · {len(df.columns)} colunas · "
          f"{time.time() - t0:.1f}s")

    saida.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSalvando {len(df):,} cursos × {len(df.columns)} colunas (gzip)...")
    df.to_csv(saida, index=False, encoding="utf-8", compression="gzip")

    tamanho_mb = saida.stat().st_size / 1024 ** 2
    print(f"\n  Concluído em {time.time() - t0:.1f}s")
    print(f"  Arquivo : {saida.resolve()}")
    print(f"  Tamanho : {tamanho_mb:.2f} MB")
    print(f"  Cursos  : {len(df):,}")
    print()
    print("Pronto. O E-XplainENADE carrega este arquivo automaticamente (streamlit run app.py).")


if __name__ == "__main__":
    main()
