"""Declarative base for the ORM models.

All ORM classes in this package inherit from :class:`BaseORM` so that
``BaseORM.metadata`` knows about every table.
"""
from sqlalchemy.orm import DeclarativeBase


class BaseORM(DeclarativeBase):
    """Shared SQLAlchemy declarative base for every ORM model."""
    pass
