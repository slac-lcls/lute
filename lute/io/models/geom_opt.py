import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, PositiveInt, validator, root_validator

from .base import TaskParameters, TemplateConfig

class OptimizePyFAIGeomParameters(TaskParameters):
    """Parameters for optimizing the geometry using PyFAI.
    
    blablabla
    """

    class Config(TaskParameters.Config):
        pass