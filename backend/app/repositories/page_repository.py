"""Repository for Page persistence operations."""

from typing import List, Optional, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.page import Page


class PageRepository:
    """Encapsulates persistent database operations for Page entities."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, page: Page) -> Page:
        """Persist a single Page entity."""
        self.session.add(page)
        self.session.flush()
        return page

    def create_many(self, pages: Sequence[Page]) -> List[Page]:
        """Persist multiple Page entities in batch."""
        page_list = list(pages)
        if not page_list:
            return []
        self.session.add_all(page_list)
        self.session.flush()
        return page_list

    def get_by_id(self, page_id: str) -> Optional[Page]:
        """Retrieve a Page by its primary identifier."""
        stmt = select(Page).where(Page.page_id == page_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_by_document(self, document_id: str) -> List[Page]:
        """Retrieve all pages for a given document ordered by page number."""
        stmt = (
            select(Page)
            .where(Page.document_id == document_id)
            .order_by(Page.page_number.asc())
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_by_document_and_page(
        self,
        document_id: str,
        page_number: int,
    ) -> Optional[Page]:
        """Retrieve a specific page by document ID and page number."""
        stmt = select(Page).where(
            Page.document_id == document_id,
            Page.page_number == page_number,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_by_document(self, document_id: str) -> int:
        """Count total pages associated with a specific document."""
        stmt = (
            select(func.count())
            .select_from(Page)
            .where(Page.document_id == document_id)
        )
        return self.session.execute(stmt).scalar_one() or 0
