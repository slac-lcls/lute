"""Generic/reusable validators for parameter models.

Functions:
    template_parameter_validator: Prepare a model which accepts template
        parameters for validation.
"""

__all__ = ["template_parameter_validator"]
__author__ = "Gabriel Dorlhiac"

from typing import Dict, Any, Optional

from pydantic import validator


def template_parameter_validator(template_params_name: str):
    """Populates a TaskParameters model with a set of validated TemplateParameters.

    This validator is intended for use with third-party Task's which use a
    templated configuration file. The validated template parameters are passed
    as a dictionary through a single parameter in the TaskParameters model. This
    dictionary is typically actually a separate pydantic BaseModel defined either
    within the TaskParameter class or externally. The other model provides the
    initial validation of each individual template parameter.

    This validator populates the TaskParameter model with the template parameters.
    It then returns `None` for the initial parameter that held these parameters.
    Returning None ensures that no attempt is made to pass the parameters on the
    command-line/when launching the third-party Task.
    """

    def _template_parameter_validator(
        cls, template_params: Optional[Any], values: Dict[str, Any]
    ) -> None:
        if template_params is not None:
            for param, value in template_params:
                values[param] = value
        return None

    return validator(template_params_name, always=True, allow_reuse=True)(
        _template_parameter_validator
    )
