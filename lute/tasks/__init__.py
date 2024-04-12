"""LUTE Tasks

Functions:
    import_task(task_name: str) -> Type[Task]: Provides conditional import of
       Task's. This prevents import conflicts as Task's may be intended to run
       in different environments.
Exceptions:
    TaskNotFoundError: Raised if
"""

from typing import Type

from .task import Task


class TaskNotFoundError(Exception):
    """Exception raised if an unrecognized Task is requested.

    The Task could be invalid (e.g. misspelled, nonexistent) or it may not have
    been registered with the `import_task` function below.
    """

    ...


def import_task(task_name: str) -> Type[Task]:
    """Conditionally imports Task's to prevent environment conflicts.

    Args:
        task_name (str): The name of the Task to import.

    Returns:
        TaskType (Type[Task]): The requested Task class.

    Raises:
        TaskNotFoundError: Raised if the requested Task is unrecognized.
            If the Task exits it may not have been registered.
    """
    if task_name == "Test":
        from .test import Test

        return Test

    if task_name == "TestSocket":
        from .test import TestSocket

        return TestSocket

    if task_name == "TestReadOutput":
        from .test import TestReadOutput

        return TestReadOutput

    if task_name == "TestWriteOutput":
        from .test import TestWriteOutput

        return TestWriteOutput

    if task_name == "FindPeaksPyAlgos":
        from .sfx_find_peaks import FindPeaksPyAlgos

        return FindPeaksPyAlgos

    if task_name == "ConcatenateStreamFiles":
        from .sfx_index import ConcatenateStreamFiles

        return ConcatenateStreamFiles

    raise TaskNotFoundError
