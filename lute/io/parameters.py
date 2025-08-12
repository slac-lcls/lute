"""Module for LUTE parameter objects.

This module contains objects that define LUTE TaskParameters. It is separate
from the pydantic model definitions included in `lute.io.models`. This allows
LUTE first-party code to run without pydantic validation. Validation is still
required to have occurred at some point to enter correct values into the database.
"""

from typing import List

__all__: List[str] = []
__author__ = "Gabriel Dorlhiac"

from typing import Set

LUTE_PARAMETER_CONFIG_KEYS: Set[str] = {
    # All Tasks
    "run_directory",
    "set_result",
    "result_from_params",
    "result_summary",
    "impl_schemas",
    # Third-party only
    "short_flags_use_eq",
    "long_flags_use_eq",
}


LUTE_PARAMETER_FIELD_ATTRS: Set[str] = {
    "flag_type",
    "rename_param",
    "description",
    "is_result",
}
