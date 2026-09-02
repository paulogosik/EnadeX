from dataclasses import dataclass
from typing import Iterator

from fastapi import Query
from psycopg2.extensions import cursor as PgCursor
from psycopg2.extras import RealDictCursor

from enade_time.api.database import get_pool


def get_db_cursor() -> Iterator[PgCursor]:
    pool = get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@dataclass
class PaginationParams:
    page: int
    page_size: int

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size


def pagination_params(
    page: int = Query(1, ge=1, description="Página (>=1)"),
    page_size: int = Query(100, ge=1, le=1000, description="Itens por página (1-1000)"),
) -> PaginationParams:
    return PaginationParams(page=page, page_size=page_size)


_FILTRO_COLUNAS = {
    "nu_ano": "nu_ano",
    "co_regiao": "co_regiao_curso",
    "co_uf": "co_uf_curso",
    "co_ies": "co_ies",
    "co_grupo": "co_grupo",
    "co_modalidade": "co_modalidade",
    "co_categad": "co_categad",
    "co_orgacad": "co_orgacad",
}


@dataclass
class FiltrosAnalise:
    nu_ano: int | None = None
    co_regiao: int | None = None
    co_uf: int | None = None
    co_ies: int | None = None
    co_grupo: int | None = None
    co_modalidade: int | None = None
    co_categad: int | None = None
    co_orgacad: int | None = None

    def to_where(self, alias: str | None = None) -> tuple[str, dict]:
        prefix = f"{alias}." if alias else ""
        fragments: list[str] = []
        params: dict = {}
        for key, col in _FILTRO_COLUNAS.items():
            value = getattr(self, key)
            if value is not None:
                fragments.append(f"AND {prefix}{col} = %({key})s")
                params[key] = value
        return (" ".join(fragments), params)


def filtros_analise(
    nu_ano: int | None = Query(None),
    co_regiao: int | None = Query(None),
    co_uf: int | None = Query(None),
    co_ies: int | None = Query(None),
    co_grupo: int | None = Query(None),
    co_modalidade: int | None = Query(None),
    co_categad: int | None = Query(None),
    co_orgacad: int | None = Query(None),
) -> FiltrosAnalise:
    return FiltrosAnalise(
        nu_ano=nu_ano,
        co_regiao=co_regiao,
        co_uf=co_uf,
        co_ies=co_ies,
        co_grupo=co_grupo,
        co_modalidade=co_modalidade,
        co_categad=co_categad,
        co_orgacad=co_orgacad,
    )
