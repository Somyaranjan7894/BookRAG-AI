"""Add processing_stage and error_message to documents table.

Revision ID: 6e82c18d9f12
Revises: 4a719f518e2b
Create Date: 2026-09-30 23:56:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6e82c18d9f12'
down_revision: Union[str, None] = '4a719f518e2b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add processing_stage column
    op.add_column(
        'documents',
        sa.Column('processing_stage', sa.String(length=64), nullable=True),
    )
    # 2. Add error_message column
    op.add_column(
        'documents',
        sa.Column('error_message', sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('documents', 'error_message')
    op.drop_column('documents', 'processing_stage')
