"""
Alembic migrations environment for Luna AI
Configures and runs database migrations
"""
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

# This is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base from our app
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import our models
from app.models import Base

# Set target_metadata to Base.metadata
target_metadata = Base.metadata

# Import our config with database URL
from app.config import settings

# Override SQLAlchemy URL with our settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.
    
    This mode doesn't require a connection to the database,
    it just creates the SQL script to run later.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.
    
    Creates a connection to the database and runs migrations directly.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


# Choose appropriate migration function based on mode
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()