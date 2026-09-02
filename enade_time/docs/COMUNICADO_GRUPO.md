# ENADE-Time Distribuído entrou no EnadeX — o que chegou e o que vem

**De:** Lucas Eduardo Tavares Costa · **Data:** 02/09/2026
**Branch:** `enade-time/migracao` → pasta **`enade_time/`** (nada fora dela foi alterado)

---

## O que é

O Projeto 4 do ecossistema (Critério B, Módulos 1 e 3): a **base longitudinal do
ENADE 2005–2021** para Computação no Norte/Nordeste (24.967 inscrições =
**582 cursos-ano**, a unidade analítica), o **experimento de processamento
paralelo** da disciplina de SPD e o **dashboard React** que exibe tudo.

## Como rodar (da raiz do EnadeX, como o educluster)

```powershell
cd C:\...\EnadeX
$env:PYTHONPATH='.'
pip install -r enade_time\requirements.txt
cd enade_time; docker compose up -d postgres; cd ..     # banco local do subprojeto
python enade_time\scripts\07_validar_banco.py            # → BASE VALIDADA (24.967)
python enade_time\enade_time_rotas.py                    # API standalone na PORTA 8002
```

Swagger: `http://localhost:8002/docs` — **19 endpoints read-only** sob
`/api/enade-time/*` (health, dimensões, análises, benchmark). Para o
`api_main.py` central: `from enade_time.enade_time_rotas import router`
(o docstring explica o ciclo de vida do pool de conexões).

Portas combinadas: 8000 = multi_enade · 8001 = educluster/E-XplainENADE ·
**8002 = enade_time**.

## O que foi entregue nesta branch

- **Convenções do repositório respeitadas:** pasta `snake_case` importável,
  `enade_time_rotas.py` com `APIRouter` prefixado (sem colisão de rotas),
  `requirements.txt`/`.gitignore` próprios, imports qualificados
  `enade_time.*` — roda da raiz com `PYTHONPATH=.`.
- **Microdados brutos (3+ GB) NÃO entram no git** — quem for rodar o ETL
  aponta `ENADE_TIME_MICRODADOS_DIR` (ver `enade_time/.env.example`). O CSV
  consolidado (5 MB) está versionado em `dados_processados/`.
- **Benchmark v2, refeito com rigor antes de migrar:** campanha oficial com
  repetições (5 suítes completas + 1 parcial preservada), duas ordens de
  submissão (crescente × LPT/Graham), máquina declarada (4 núcleos físicos /
  8 lógicos), cache aquecido e instrumentação de CPU/disco por execução.
  Speedup/eficiência têm **uma única definição, em views SQL** — a API só lê.
  Resultado-chave: melhor configuração **6 workers·crescente, 2,701× mediana**;
  a perda é decomposta em granularidade+escalonamento / contenção de CPU /
  overhead (`docs/RESULTADOS_BENCHMARK.md`, gerado do banco).
- **Qualidade:** 59 testes pytest (`enade_time/tests`, pulados sem o Postgres
  local), regressão "golden" das rotas, e `docs/geradores/verificar_numeros.py`
  garante que **nenhum número** do documento/slides existe sem linha no banco.
- Correção documentada de um bug real: o comparativo antigo usava "o sequencial
  mais recente" como baseline e misturou rodadas (`docs/DESIGN_LOG.md`, D13).

## O que ainda vem (fases seguintes, já planejadas)

1. **Publicação no Supabase** com os nomes na convenção do grupo:
   `tbl_enade_time_fato` (24.967 linhas, consumível por `consultar_dados`),
   `tbl_enade_time_dim_*` e as views de benchmark.
2. **`v_enade_time_curso_ano` (582 linhas)** — o *swap point* combinado com o
   E-XplainENADE, com os MESMOS nomes de coluna do `load_raw()` do JP
   (`NU_ANO, CO_CURSO, NT_GER, NT_FG, NT_CE, QT_ALUNOS, CO_GRUPO, CO_REGIAO,
   TP_CATEGAD_BIN`): extensão longitudinal 2005–2017 para o recorte comum.
3. Router 100 % autocontido (pool inicializado na 1ª requisição) + proposta de
   `api_main.py` central de referência para o líder.
4. Medição do **custo de publicar na nuvem** (COPY local × COPY via pooler ×
   upsert PostgREST) com o mesmo instrumental do benchmark.
5. PR final para a `main` + evidências/screenshots atualizados.

## Pedidos ao grupo

- **Segurança (já enviado ao líder, urgente):** revogar `EXECUTE` da RPC
  `truncar_tabela` para `anon`/`authenticated` no Supabase e revisar o modelo
  de chaves — detalhes em `enade_time/docs/COMUNICADO_SEGURANCA.md`.
- **JP:** confirmar o contrato do swap point (nomes acima; a diferença de
  definição de `QT_ALUNOS` — inscritos × presentes — vai documentada na view).
- **Líder:** compartilhar o `DEVELOPMENT.md`/atas (as decisões citadas no
  código do E-XplainENADE estão fora do repositório) — não bloqueia, mas evita
  conflito de nomenclatura na publicação.

Dúvidas ou algo que interfira no subprojeto de vocês: me chamem. Nada fora de
`enade_time/` foi tocado, e tudo o que está aqui roda de ponta a ponta hoje.
