"""Specifies custom exceptions defined for IO problems.

Exceptions:
    ElogFileFormatError: Raised if an attachment is specified in an incorrect
        format.
"""

__all__ = ["ElogFileFormatError"]
__author__ = "Gabriel Dorlhiac"


class ElogFileFormatError(Exception):
    """Raised when an eLog attachment is specified in an invalid format."""

    ...
