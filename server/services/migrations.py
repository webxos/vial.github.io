from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


def upgrade():
    op.create_table(
        "wallets",
        sa.Column("user_id", sa.String, primary_key=True),
        sa.Column("balance", sa.Float, nullable=False),
        sa.Column("network_id", sa.String, nullable=False)
    )


def downgrade():
    op.drop_table("wallets")
