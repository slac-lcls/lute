"""Models for running the Cheetah Task

Classes:
    RunCheetahParameters(ThirdPartyParameters): Parameters to run Cheetah.
    More information on Cheetah: https://github.com/omdevteam/om
"""

__all__ = ["RunCheetahParameters"]


from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, root_validator

from lute.io.models.base import (
    ThirdPartyParameters,
    TemplateConfig,
    validator,
)
from lute.io.models.sfx_find_peaks import SZCompressorParameters
from lute.io.models.validators import template_parameter_validator


class RunCheetahParameters(ThirdPartyParameters):
    """Parameters for running OM Cheetah Task."""

    class CheetahSubconfigParameters(BaseModel):
        """Parameters for OM Cheetah itself."""

        class OmParameters(BaseModel):
            """Parameters for OM layers"""

            processing_layer: Optional[str] = Field(
                "",
                flag_type="",
            )

        class DataRetrievalLayerParameters(BaseModel):
            """Parameters for the data retrieving layer."""

            class DataSourcesParameters(BaseModel):
                """Parameters for the data retrieval layer data sources subfields."""

                class DataSourceDetectorDataParameters(BaseModel):
                    """Data retrieval layer, datasource detector data parameters."""

                    psana_name: Optional[str] = Field("epix10k2M", flat_type="")
                    calibration: Optional[str] = Field("true", flat_type="")

                class DataSourceDetectorDistanceParameters(BaseModel):
                    """Data retrieval layer, datasource detector distance parameters."""

                    type: Optional[str] = Field("EpicsVariablePsana", flat_type="")
                    psana_name: Optional[str] = Field("detector_z", flat_type="")
                    value: Optional[str] = Field("DetectorDistanceValue", flat_type="")

            psana_calibration_directory: Optional[str] = Field(
                "/sdf/data/lcls/ds/xpp/xpptut15/calib/",
                description="Location of the psana1 calibration directory.",
            )
            data_sources: Optional[DataSourcesParameters] = Field(
                None, description="Configuration for data source retrieval."
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

        class DataCompressionParameters(BaseModel):
            run_compression: bool = Field(
                False,
                description="Whether to enable compression.",
            )
            backend: Optional[Literal["roibinsz"]] = Field(
                None,
                description=(
                    "The compression backend to use. Currently only RoiBinSz "
                    "with libpressio is available."
                ),
            )
            compression_parameters: Optional[SZCompressorParameters] = Field(
                None,
                description="The actual compression parameters. Varies by backend.",
            )

            @root_validator(pre=False)
            def check_backend_matches_parameters(
                cls, values: Dict[str, Any]
            ) -> Dict[str, Any]:
                if values["backend"] == "roibinsz":
                    if not isinstance(
                        values["compression_parameters"], SZCompressorParameters
                    ):
                        raise ValueError(
                            "For the libpressio RoiBinSz compression backend you must "
                            "use SZCompressorParameters for `compression_parameters`."
                        )
                return values

        om: Optional[OmParameters] = Field(None, description="Global options for OM.")
        data_retrieval_layer: Optional[DataRetrievalLayerParameters] = Field(
            None,
            description="Options for configuring the data retrieval.",
        )
        cheetah: Optional[CheetahParameters] = Field(
            None, description="Cheetah specific parameters (like output directories)."
        )
        crystallography: Optional[CrystallographyParameters] = Field(
            None, description="Generic crystallography parameters, like geometry file."
        )
        peakfinder8_peak_detection: Optional[Peakfinder8PeakDetectionParameters] = (
            Field(
                None,
                description="Peakfinder8 peak finding algorithm specific parameters.",
            )
        )
        compression: Optional[DataCompressionParameters] = Field(
            None, description="Data compression parameters."
        )

    _set_cheetah_template_parameters = template_parameter_validator("cheetah_subconfig")

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
    cheetah_subconfig: Optional[CheetahSubconfigParameters] = Field(
        None,
        description="The parameter set for the OM configuration YAML.",
        flag_type="",  # Does nothing since None by time it's seen by Task
    )
    lute_template_cfg: TemplateConfig = Field(
        TemplateConfig(
            template_name="cheetah_template.yaml",
            output_path="/sdf/scratch/users/k/kmecseki/cheetah/cheetah_testconfig.yaml",
        ),
        description="Template rendering configuration",
    )

    @validator("lute_template_cfg", always=True)
    def update_output_path(
        cls, lute_template_cfg: TemplateConfig, values: Dict[str, Any]
    ) -> TemplateConfig:
        if lute_template_cfg.output_path == "":
            lute_template_cfg.output_path = values["om_config"]
        return lute_template_cfg
