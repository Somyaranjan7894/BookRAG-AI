"""add_pgvector_and_chunk_embedding

Revision ID: 4a719f518e2b
Revises: 3f646c4705d5
Create Date: 2026-09-30 23:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.core.config import settings

# revision identifiers, used by Alembic.
revision: str = '4a719f518e2b'
down_revision: Union[str, Sequence[str], None] = '3f646c4705d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable vector extension, add embedding column to chunks, and create HNSW index."""
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Add embedding VECTOR(384) column to chunks table
    op.add_column(
        'chunks',
        sa.Column(
            'embedding',
            Vector(settings.EMBEDDING_DIMENSION),
            nullable=True,
        ),
    )

    # 3. Create HNSW vector similarity index for cosine similarity
    op.create_index(
        'ix_chunks_embedding_hnsw',
        'chunks',
        ['embedding'],
        unique=False,
        postgresql_using='hnsw',
        postgresql_with={'m': 16, 'ef_construction': 64},
        postgresql_ops={'embedding': 'vector_cosine_ops'},
    )


def downgrade() -> None:
    """Drop HNSW index, drop embedding column, and drop vector extension."""
    op.drop_index(
        'ix_chunks_embedding_hnsw',
        table_name='chunks',
        postgresql_using='hnsw',
    )
    op.drop_column('chunks', 'embedding')
    op.execute("DROP EXTENSION IF EXISTS vector;")
