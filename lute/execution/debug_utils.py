"""Functions to assist in debugging execution of LUTE.

Functions:
    LUTE_DEBUG_EXIT(env_var: str, str_dump: Optional[str]): Exits the program if
        the provided `env_var` is set. Optionally, also prints a message if
        provided.

Exceptions:
    ValidationError: Error raised by pydantic during data validation. (From
        Pydantic)
"""

__all__ = ["LUTE_DEBUG_EXIT"]
__author__ = "Gabriel Dorlhiac"

import os
import sys
import types
from typing import Optional


def _stack_inspect(msg: str, str_dump: Optional[str] = None) -> None:
    import inspect

    curr_frame: Optional[types.FrameType] = inspect.currentframe()
    frame: Optional[types.FrameType]
    if curr_frame:
        frame = curr_frame.f_back
        if frame:
            frame = frame.f_back  # Go back two stack frames...
    else:
        frame = None
    if frame:
        file_name: str = frame.f_code.co_filename
        line_no: int = frame.f_lineno
        msg = f"{msg} {file_name}, line: {line_no}"
    else:
        msg = f"{msg} Stack frame not retrievable..."
    if str_dump is not None:
        msg = f"{msg}\n{str_dump}"

    print(msg, flush=True)


def LUTE_DEBUG_EXIT(env_var: str, str_dump: Optional[str] = None) -> None:
    if os.getenv(env_var, None):
        msg: str = "LUTE_DEBUG_EXIT -"
        _stack_inspect(msg, str_dump)
        sys.exit(0)


def LUTE_DEBUG_PAUSE(env_var: str, str_dump: Optional[str] = None) -> None:
    # Need custom signal handlers to implement resume
    if os.getenv(env_var, None):
        import signal

        msg: str = "LUTE_DEBUG_PAUSE -"
        _stack_inspect(msg, str_dump)
        signal.pause()
