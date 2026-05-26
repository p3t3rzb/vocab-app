"""Declarative base for the ORM models.

All ORM classes in this package inherit from :class:`Base` so that
``Base.metadata`` knows about every table.
"""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared SQLAlchemy declarative base for every ORM model."""
    pass
