"""Utility functions for Teams bot."""

import html
import re


def safe_get_nested(data: dict, *keys, default=""):
    """Safely navigate nested dictionaries."""
    current = data
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return default
        if current is None:
            return default
    return current


def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", "", text)
    clean = html.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean
