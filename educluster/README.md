# EduCluster

Modulo do ecossistema EnadeX. Descoberta de perfis academicos latentes nos microdados do ENADE 2021.

Projeto irmao de `multi_enade/`, com pipeline e dependencias proprias. Nao modifica nenhum arquivo fora desta pasta.

## Estrutura

```
educluster/
  educluster_config.py       constantes, dimensoes do arq4, features, reprodutibilidade
  educluster_dados.py        carga plugavel: microdados locais ou Supabase
  educluster_preparo.py      limpeza da escala, derivacao de area, agregacao por curso
  educluster_endpoints.py    API FastAPI
  educluster_interpretacao.py  descricao automatica dos perfis via Gemini, com cache
  modelos/
    mdl_curso_percepcao/       A6 e A7, mais estabilidade por reamostragem
    mdl_perfil_desempenho/     A1: perfis de desempenho no nivel do estudante
    mdl_calibracao_prova/      A3: percepcao da prova versus desempenho
    mdl_situacao_discursiva/   A4: desistencia versus erro conceitual
    mdl_dimensao_desempenho/   A8: qual dimensao percebida se associa ao desempenho
```

## Analises implementadas

**A6, perfis de curso (analise central).** Cruza arq3 e arq4 agregados por `CO_CURSO`, unico join permitido pela versao LGPD dos microdados. Features: `NT_FG`, `NT_CE`, taxa de presenca, e as tres dimensoes percebidas do arq4 (ODP, OPORT, INFRA). Base de 3.832 cursos com no minimo 20 respondentes em cada arquivo, representando 316.232 estudantes em 30 areas.

**A1, perfis de desempenho.** Nivel do estudante, apenas arq3. Espaco padrao `objetivo_discursivo` (`NT_OBJ_FG`, `NT_DIS_FG`, `NT_OBJ_CE`, `NT_DIS_CE`), que substitui o trio colinear usado no TCC 1. Os espacos `trio` e `par` seguem disponiveis para comparacao.

## Decisoes de tratamento

- **Escala do arq4**: os codigos 7 (nao sei responder) e 8 (nao se aplica) viram nulo. O codigo 6 e resposta valida (concordo totalmente) e e preservado
- **Area do curso**: derivada de `DS_VT_GAB_OCE_FIN`, que tem 30 valores e mapeia 1 para 1 com o curso. Recupera o recorte por area sem depender do arq1
- **Reprodutibilidade**: `random_state=42` e `n_init=10` em todo K-Means, `sample_size` explicito no silhouette do nivel estudante

## Como rodar

Instalar dependencias e executar da raiz do repositorio (`EnadeX/`):

```
pip install -r educluster/requirements.txt
PYTHONPATH=. python -m educluster.modelos.mdl_curso_percepcao.modelo_curso_percepcao
PYTHONPATH=. python -m educluster.modelos.mdl_perfil_desempenho.modelo_perfil_desempenho
```

Subir a API:

```
PYTHONPATH=. python -m educluster.educluster_endpoints
```

## Fonte de dados

O parametro `origem` aceita `local` (le os `.txt` de `microdados/`, com cache em parquet) ou `supabase` (usa `consultar_dados` de `util/util_db.py`). O padrao e `local` porque o volume das tabelas no Supabase ainda nao foi conferido.

## Rotas

Prefixo `/api/educluster`. Organizadas por recurso e granularidade, nao por analise.

| Rota | Analise | Retorno |
|---|---|---|
| `GET /cursos` | A6 | Colecao com cluster e PCA. Filtros `area` e `cluster` |
| `GET /cursos/clusters` | A6 | Perfil de cada cluster e metricas |
| `GET /cursos/discrepantes` | A7 | Cursos que destoam dos pares da area |
| `GET /cursos/{co_curso}` | A6 | Detalhe de um curso |
| `GET /estudantes/clusters` | A1 | Perfil de cada cluster de desempenho |
| `GET /estudantes/amostra` | A1 | Amostra reprodutivel para plotagem |
| `GET /areas` | catalogo | As 30 areas com agregados |
| `GET /dimensoes` | catalogo | ODP, OPORT e INFRA com seus itens |
| `GET /espacos-desempenho` | catalogo | Espacos de variaveis da A1 |

Parametros gerais: `origem` (`local` ou `supabase`), `n_minimo`, `k` (reclusteriza em memoria).

A A1 nao expoe colecao completa: seriam 51,2 MB para 354.899 estudantes. Expoe resumo e amostra.

Status: 200 valido, 404 curso ausente, 422 parametro fora da faixa, 503 fonte indisponivel.
