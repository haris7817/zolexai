"""Alembic environment.

Two things differ from the generated default and both matter:

  * The URL comes from `DATABASE_URL`, never from `alembic.ini`, so no
    credential is ever committed.
  * Migrations run through the async driver the application itself uses, so a
    migration cannot pass here and fail at runtime because of a dialect
    difference.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlalchemy.pool import NullPool

from app.core.config import settings

# Importing the models package is what populates Base.metadata. A model not
# reachable from `app.models` is invisible to autogenerate.
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# DATABASE_URL is the default, but an explicitly-supplied url wins.
#
# That ordering matters twice: it lets an operator target a specific database
# for a one-off run (`alembic -x` or a programmatic Config) without rewriting
# the environment, and it is what allows the migration parity test to upgrade a
# throwaway database instead of whichever one the process is pointed at.
if not config.get_main_option("sqlalchemy.url", None):
    config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def _configure(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Detects a column whose type changed, which autogenerate ignores by
        # default — a silent no-op migration is worse than a noisy one.
        compare_type=True,
        compare_server_default=True,
        # Constraint names come from the metadata naming convention, so Alembic
        # can generate a DROP for something PostgreSQL named itself.
        render_as_batch=False,
        include_schemas=False,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    # NullPool: a migration run is a short-lived process, and a pool would keep
    # connections open after the work is done.
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
