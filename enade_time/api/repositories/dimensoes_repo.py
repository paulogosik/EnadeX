from psycopg2.extensions import cursor as PgCursor


def listar_regioes(cursor: PgCursor) -> list[dict]:
    cursor.execute(
        "SELECT co_regiao, nome FROM dim_regiao ORDER BY co_regiao"
    )
    return list(cursor.fetchall())


def listar_ufs(cursor: PgCursor, co_regiao: int | None = None) -> list[dict]:
    if co_regiao is None:
        cursor.execute(
            "SELECT co_uf, sigla, nome, co_regiao FROM dim_uf "
            "ORDER BY co_regiao, sigla"
        )
    else:
        cursor.execute(
            "SELECT co_uf, sigla, nome, co_regiao FROM dim_uf "
            "WHERE co_regiao = %(co_regiao)s ORDER BY sigla",
            {"co_regiao": co_regiao},
        )
    return list(cursor.fetchall())


def listar_anos(cursor: PgCursor) -> list[dict]:
    cursor.execute("SELECT co_ano, edicao FROM dim_ano ORDER BY co_ano")
    return list(cursor.fetchall())


def listar_grupos(cursor: PgCursor) -> list[dict]:
    cursor.execute("SELECT co_grupo, nome FROM dim_grupo ORDER BY co_grupo")
    return list(cursor.fetchall())


def listar_modalidades(cursor: PgCursor) -> list[dict]:
    cursor.execute(
        "SELECT co_modalidade, nome FROM dim_modalidade ORDER BY co_modalidade"
    )
    return list(cursor.fetchall())


def listar_categorias_adm(cursor: PgCursor) -> list[dict]:
    cursor.execute(
        "SELECT co_categad, nome FROM dim_categoria_adm ORDER BY co_categad"
    )
    return list(cursor.fetchall())


def listar_organizacoes_academicas(cursor: PgCursor) -> list[dict]:
    cursor.execute(
        "SELECT co_orgacad, nome FROM dim_organizacao_acad ORDER BY co_orgacad"
    )
    return list(cursor.fetchall())
