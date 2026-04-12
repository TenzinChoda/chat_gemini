"""Application configuration (paths, models). Load from environment / .env in production."""

from config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
