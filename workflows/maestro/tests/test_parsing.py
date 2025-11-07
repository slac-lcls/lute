from typing import List

from maestro._maestro._maestro import JobParameters, JobStep, TriggerRule
from maestro.parser import load_lute_dag_str

SIMPLE_DAG: str = """
!LUTE_DAG
task_name: "Tester"
slurm_params: ""
next:
- task_name: "SocketTester"
  slurm_params: ""
  next: []
"""

SIMPLE_PARALLEL_DAG: str = """
!LUTE_DAG
- task_name: "Tester"
  slurm_params: ""
  next:
  - task_name: "SocketTester"
    slurm_params: ""
    next: []
- task_name: "Tester"
  slurm_params: ""
  next:
  - task_name: "SocketTester"
    slurm_params: ""
    next: []
"""


SIMPLE_TRIGGER_DAG: str = """
!LUTE_DAG
- !ALWAYS
  task_name: "Tester"
  slurm_params: ""
  next:
  - !ALL_FAILED
    task_name: "BinaryErrTester"
    slurm_params: ""
    next: []
  - task_name: "SocketTester"
    slurm_params: ""
    next: []
  - !ALWAYS
    task_name: "WriteTester"
    slurm_params: ""
    next:
    - task_name: "ReadTester"
      slurm_params: ""
      next: []
"""


SIMPLE_BRANCH_DAG: str = """
!LUTE_DAG
- task_name: "Tester"
  slurm_params: ""
  next:
  - !ALL_FAILED
    task_name: "SocketTester"
    slurm_params: ""
    next: []
- task_name: "Tester"
  slurm_params: ""
  next:
  - !branch_daq2
    daq2:
      task_name: "SocketTester"
      slurm_params: ""
      next: []
    daq1:
      task_name: "WriteTester"
      slurm_params: ""
      next:
      - task_name: "ReadTester"
        slurm_params: ""
        next: []
"""


def print_wf(wf: List[JobStep]):
    for job in wf:
        print(f"{job.managed_task_name} triggers on {job.trigger_rule}")
        print_wf(job.next)


def is_equal(steps_0: List[JobStep], steps_1: List[JobStep]) -> bool:
    if len(steps_0) != len(steps_1):
        return False
    else:
        all_equal: bool = True
        for i in range(len(steps_0)):
            if steps_0[i].managed_task_name != steps_1[i].managed_task_name:
                all_equal = False
                break
            if steps_0[i].trigger_rule != steps_1[i].trigger_rule:
                all_equal = False
                break
            if (
                steps_0[i].parameters.lute_location
                != steps_1[i].parameters.lute_location
            ):
                all_equal = False
                break
            if steps_0[i].parameters.config_file != steps_1[i].parameters.config_file:
                all_equal = False
                break
            if steps_0[i].parameters.debug != steps_1[i].parameters.debug:
                all_equal = False
                break
            if steps_0[i].extra_parameters != steps_1[i].extra_parameters:
                all_equal = False
                break
            all_equal = is_equal(steps_0[i].next, steps_1[i].next)
        return all_equal


class TestParsing:
    # These values don't matter since we are only testing
    lute_location: str = "/path/to/lute"
    config_file: str = "/path/to/config"
    debug: bool = True
    gen_params: JobParameters = JobParameters(lute_location, config_file, debug)

    def test_simple(self):
        job_steps: List[JobStep] = [
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [],
                    )
                ],
            )
        ]
        wf_defn: List[JobStep] = load_lute_dag_str(
            workflow_str=SIMPLE_DAG,
            lute_location=TestParsing.lute_location,
            config_file=TestParsing.config_file,
            debug=TestParsing.debug,
            branch_conditions={"daq2": True},
        )
        assert is_equal(wf_defn, job_steps)  # wf_defn == job_steps

    def test_parallel(self):
        job_steps: List[JobStep] = [
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [],
                    )
                ],
            ),
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [],
                    )
                ],
            ),
        ]
        wf_defn: List[JobStep] = load_lute_dag_str(
            workflow_str=SIMPLE_PARALLEL_DAG,
            lute_location=TestParsing.lute_location,
            config_file=TestParsing.config_file,
            debug=TestParsing.debug,
            branch_conditions={"daq2": True},
        )
        assert is_equal(wf_defn, job_steps)

    def test_trigger(self):
        job_steps: List[JobStep] = [
            JobStep(
                "Tester",
                TriggerRule.ALWAYS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "BinaryErrTester",
                        TriggerRule.ALL_FAILED,
                        TestParsing.gen_params,
                        "",
                        [],
                    ),
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [],
                    ),
                    JobStep(
                        "WriteTester",
                        TriggerRule.ALWAYS,
                        TestParsing.gen_params,
                        "",
                        [
                            JobStep(
                                "ReadTester",
                                TriggerRule.ALL_SUCCESS,
                                TestParsing.gen_params,
                                "",
                                [],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        wf_defn: List[JobStep] = load_lute_dag_str(
            workflow_str=SIMPLE_TRIGGER_DAG,
            lute_location=TestParsing.lute_location,
            config_file=TestParsing.config_file,
            debug=TestParsing.debug,
            branch_conditions={"daq2": True},
        )
        assert is_equal(wf_defn, job_steps)

    def test_branch_daq2(self):
        job_steps: List[JobStep] = [
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_FAILED,
                        TestParsing.gen_params,
                        "",
                        [],
                    ),
                ],
            ),
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [],
                    ),
                ],
            ),
        ]
        wf_defn: List[JobStep] = load_lute_dag_str(
            workflow_str=SIMPLE_BRANCH_DAG,
            lute_location=TestParsing.lute_location,
            config_file=TestParsing.config_file,
            debug=TestParsing.debug,
            branch_conditions={"daq2": True},
        )
        assert is_equal(wf_defn, job_steps)

    def test_branch_daq1(self):
        job_steps: List[JobStep] = [
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "SocketTester",
                        TriggerRule.ALL_FAILED,
                        TestParsing.gen_params,
                        "",
                        [],
                    ),
                ],
            ),
            JobStep(
                "Tester",
                TriggerRule.ALL_SUCCESS,
                TestParsing.gen_params,
                "",
                [
                    JobStep(
                        "WriteTester",
                        TriggerRule.ALL_SUCCESS,
                        TestParsing.gen_params,
                        "",
                        [
                            JobStep(
                                "ReadTester",
                                TriggerRule.ALL_SUCCESS,
                                TestParsing.gen_params,
                                "",
                                [],
                            ),
                        ],
                    ),
                ],
            ),
        ]
        wf_defn: List[JobStep] = load_lute_dag_str(
            workflow_str=SIMPLE_BRANCH_DAG,
            lute_location=TestParsing.lute_location,
            config_file=TestParsing.config_file,
            debug=TestParsing.debug,
            branch_conditions={"daq2": False},
        )
        assert is_equal(wf_defn, job_steps)
