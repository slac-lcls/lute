"""Models for running the Cheetah Task

Classes:
    RunCheetahParameters(ThirdPartyParameters): Parameters to run Cheetah.
    More information on Cheetah: https://github.com/omdevteam/om
"""

__all__ = ["RunCheetahParameters"]


from pydantic import BaseModel, Field
from typing import Any, Dict
from lute.io.models.base import (
    ThirdPartyParameters,
    TemplateConfig,
    validator,
    Optional,
)


class RunCheetahParameters(ThirdPartyParameters):
    """Parameters for running OM Cheetah Task."""

    executable: str = Field(
        "/sdf/group/lcls/ds/tools/openmpi-5.0.6/bin/mpirun",
        description="MPI executable.",
        flag_type="",
    )
    n_processes: int = Field(
        4,
        description="Number of processes to launch.",
        flag_type="-",
        rename_param="n",
    )
    om_executable: str = Field(
        "/sdf/group/lcls/ds/tools/om/om/om-071725/bin/om_monitor.py",
        description="OM Cheetah binary.",
        flag_type="",
    )
    om_config: str = Field(
        "/sdf/scratch/users/k/kmecseki/cheetah/test.yaml",
        description="Config file for OM Cheetah.",
        flag_type="-",
        rename_param="c",
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cheetah_template.yaml",
            output_path="/sdf/scratch/users/k/kmecseki/cheetah/cheetah_testconfig.yaml",
        ),
        description="Template rendering configuration",
    )

    class CheetahSubconfigParameters(BaseModel):
        """Parameters for OM Cheetah itself."""

        class Config(BaseModel.Config):
            extra: str = "allow"

        class OmParameters(BaseModel):
            """Parameters for OM layers"""

            processing_layer: Optional[str] = Field(
                "",
                flag_type="",
            )

        class DataRetrievalLayerParameters(BaseModel):
            """Parameters for the data retrieving layer."""

            psana_calibration_directory: Optional[str] = Field(
                "/sdf/data/lcls/ds/xpp/xpptut15/calib/",
                flag_type="",
            )

        class CheetahParameters(BaseModel):
            """Parameters for Cheetah."""

            processed_directory: Optional[str] = Field(
                "/sdf/data/lcls/ds/xpp/xpptut15/results",
                flag_type="",
            )
            processed_filename_prefix: Optional[str] = Field(
                "xpptut15-092",
                flag_type="",
            )
            class_sums_filename_prefix: Optional[str] = Field(
                "92",
                flag_type="",
            )

        class CrystallographyParameters(BaseModel):
            """Parameters for Crystallography."""

            geometry_file: Optional[str] = Field(
                "/sdf/scratch/users/k/kmecseki/cheetah/test.geom",
                flag_type="",
            )
            responding_url: Optional[str] = Field(
                "",
                flag_type="",
            )

        class Peakfinder8PeakDetectionParameters(BaseModel):
            """Parameters for peak finder 8 peak detection."""

            adc_threshold: Optional[float] = Field(
                100.0,
                flag_type="",
            )
            minimum_snr: Optional[float] = Field(
                5.0,
                flag_type="",
            )
            min_pixel_count: Optional[int] = Field(
                1,
                flag_type="",
            )
            max_pixel_count: Optional[int] = Field(
                30,
                flag_type="",
            )
            local_bg_radius: Optional[int] = Field(
                4,
                flag_type="",
            )
            bad_pixel_map_filename: Optional[str] = Field(
                "",
                flag_type="",
            )
            min_res: Optional[int] = Field(
                80,
                flag_type="",
            )
            max_res: Optional[int] = Field(
                800,
                flag_type="",
            )

    @validator("lute_template_cfg", always=True)
    def update_output_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["om_config"]
        return lute_template_cfg
