from psycopg2.pool import ThreadedConnectionPool

from enade_time.api.settings import Settings

_pool: ThreadedConnectionPool | None = None


def init_pool(settings: Settings) -> None:
    global _pool
    if _pool is not None:
        return
    _pool = ThreadedConnectionPool(
        minconn=settings.pool_minconn,
        maxconn=settings.pool_maxconn,
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
    )


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None


def get_pool() -> ThreadedConnectionPool:
    if _pool is None:
        raise RuntimeError("Connection pool not initialized")
    return _pool
