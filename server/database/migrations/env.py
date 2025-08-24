```python
from alembic import context
from sqlalchemy import engine_from_config, pool
from ..database.base import Base
from ..database.gibs_models import GIBSMetadata

config = context.config
connectable = engine_from_config(config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool)
Base.metadata.bind = connectable

with connectable.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata
    )

    with context.begin_transaction():
        context.run_migrations()
```
