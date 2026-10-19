import copy
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest

from lute.io.config import resolve_run_group_directives


class TestRunGroupResolution:
    def test_applies_to_range_matching(self) -> None:
        """Test inclusive range matching [0, 40]."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@db:current",
            },
        }
        task_config: Dict[str, int] = {
            "param1": 10,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry") as mock_db:
            mock_db.return_value = 99

            res_inside: Dict[str, int] = resolve_run_group_directives(
                "MyTask", task_config.copy(), global_groups, "/tmp", current_run=20
            )
            assert res_inside["param1"] == 99
            mock_db.assert_called_once_with(
                db_dir="/tmp", task_name="MyTask", param="param1", for_run=20
            )

            mock_db.reset_mock()
            res_outside: Dict[str, int] = resolve_run_group_directives(
                "MyTask", task_config.copy(), global_groups, "/tmp", current_run=50
            )
            assert res_outside["param1"] == 10
            mock_db.assert_not_called()

    def test_applies_to_discrete_runs_list(self) -> None:
        """Test discrete runs list matching [39, 50, 60]."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "SPECIAL": {
                "applies_to": {
                    "runs": [39, 50, 60],
                },
                "default": "@db:current",
            },
        }
        task_config: Dict[str, int] = {
            "param1": 5,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=42):
            res_match: Dict[str, int] = resolve_run_group_directives(
                "MyTask", task_config.copy(), global_groups, "/tmp", current_run=50
            )
            assert res_match["param1"] == 42

            res_no_match: Dict[str, int] = resolve_run_group_directives(
                "MyTask", task_config.copy(), global_groups, "/tmp", current_run=40
            )
            assert res_no_match["param1"] == 5

    def test_all_three_directives(self) -> None:
        """Test all directives can mix and match.

        These are:
        - @db:current
        - @db:run:NNN
        - @literal
        """
        # fmt: off
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "MIXED": {
                    "applies_to": {
                        "range": [0, 100],
                    },
                    "default": "@literal",
                    "p_current": "@db:current",
                    "p_run40": "@db:run:40",
                    "p_literal": "@literal",
                },
            },
            "p_current": 1,
            "p_run40": 2,
            "p_literal": 3,
            "p_unspecified": 4,  # Uses group default (@literal)
        }
        # fmt: on

        def mock_db_lookup(
            db_dir: str, task_name: str, param: int, for_run: int
        ) -> Optional[int]:
            if param == "p_current" and for_run == 15:
                return 100
            elif param == "p_run40" and for_run == 40:
                return 200
            return None

        with patch("lute.io.config.read_latest_db_entry", side_effect=mock_db_lookup):
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", task_config, global_groups={}, work_dir="/tmp", current_run=15
            )

            # fmt: off
            assert res["p_current"] == 100    # Resolved from run 15
            assert res["p_run40"] == 200      # Resolved specifically from run 40
            assert res["p_literal"] == 3      # Kept literal
            assert res["p_unspecified"] == 4  # Kept literal via group default
            # fmt: on

    def test_task_group_overrides_global_group(self) -> None:
        """Test per-Task RUN_GROUPS overriding top-level global RUN_GROUPS."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@literal",
                # Note: In general, you wouldn't really do this. Only the default.
                #       This is just for testing/
                "ROIs": "@literal",
            }
        }
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "DARK": {
                    "ROIs": "@db:run:40",  # Task-specific override!
                },
            },
            "ROIs": "literal_roi",
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value="db_roi"):
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", task_config, global_groups, work_dir="/tmp", current_run=10
            )
            assert res["ROIs"] == "db_roi"

    def test_swept_task_name_suffix_stripping(self) -> None:
        """Test handling of parameter sweeps.

        The parameter sweep mechanism adds a suffix to the generated configurations.
        When doing any database lookups, that suffix must be stripped.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "ALL": {
                "applies_to": {
                    "range": [0, 100],
                },
                "default": "@db:current",
            },
        }
        task_config: Dict[str, int] = {
            "param1": 10,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry") as mock_db:
            mock_db.return_value = 50
            resolve_run_group_directives(
                "SubmitSMD_0",
                task_config,
                global_groups,
                work_dir="/tmp",
                current_run=5,
            )

            # Should look up 'SubmitSMD' in DB, NOT 'SubmitSMD_0'
            mock_db.assert_called_once_with(
                db_dir="/tmp", task_name="SubmitSMD", param="param1", for_run=5
            )


class TestRunGroupEdgeCases:
    def test_missing_db_entry_fallback(self) -> None:
        """Test that missing DB entries log a warning and fall back to YAML literal.

        NOTE: Should use the fallback -- Pydantic fails hard if there are other errors.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@db:current",
            },
        }
        task_config: Dict[str, str] = {
            "param1": "yaml_default",
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=None):
            with pytest.warns(UserWarning, match="No DB entry found"):
                res: Dict[str, str] = resolve_run_group_directives(
                    "MyTask", task_config, global_groups, "/tmp", current_run=10
                )
            # Should fall back to literal value in YAML
            assert res["param1"] == "yaml_default"

    def test_missing_or_invalid_run_number(self) -> None:
        """Test behavior when RUN env var or header run is missing/non-numeric.

        NOTE: Should use fallback to YAML. Pydantic will fail hard if there are other
              errors.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@db:current",
            },
        }
        task_config: Dict[str, int] = {
            "param1": 10,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry") as mock_db:
            # current_run = None
            res_none: Dict[str, int] = resolve_run_group_directives(
                "MyTask", task_config.copy(), global_groups, "/tmp", current_run=None
            )
            assert res_none["param1"] == 10

            # current_run = "invalid"
            res_inv: Dict[str, int] = resolve_run_group_directives(
                "MyTask",
                task_config.copy(),
                global_groups,
                "/tmp",
                current_run="invalid",
            )
            assert res_inv["param1"] == 10

            mock_db.assert_not_called()

    def test_malformed_directive_string(self) -> None:
        """Test handling of invalid @db: string directives.

        NOTE: Should warn -- but uses fallback. Pydantic will fail hard if there are
              other errors.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@db:run:not_an_int",  # Malformed run number
            },
        }
        task_config: Dict[str, str] = {
            "param1": "fallback",
        }
        # fmt: on

        with pytest.warns(UserWarning, match="Invalid run directive format"):
            res: Dict[str, str] = resolve_run_group_directives(
                "MyTask", task_config, global_groups, "/tmp", current_run=10
            )

        assert res["param1"] == "fallback"


class TestRunGroupPrecedence:
    """Test the precendence and conflict rules for RUN_GROUPS.

    The rules are:
    - Per-Task, the set of RUN_GROUPS is the union of the global and Task definitions
    - A Task block definition always trumps a global definition
      - With one exception - a more specific global group range still trumps the Task
    - A list of runs trumps a range of runs for defining included runs for the group
    - A narrower list of runs trumps a longer list of runs
    - A narrower range of runs trumps a longer range of runs
    """

    def test_union_of_global_and_task_groups(self) -> None:
        """Verify both global and Task-level groups are used."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                # Set a default global policy
                "default": "@db:current",
            },
        }
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "SPECIAL": {
                    "applies_to": {
                        "runs": [100],
                    },
                    "default": "@db:run:7",
                }
            },
            "param1": 1,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=99) as mock_db:
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", copy.deepcopy(task_config), global_groups, "/tmp", 20
            )
            assert res["param1"] == 99
            assert mock_db.call_args.kwargs["for_run"] == 20  # via global DARK

            res = resolve_run_group_directives(
                "MyTask", copy.deepcopy(task_config), global_groups, "/tmp", 100
            )
            assert res["param1"] == 99
            assert mock_db.call_args.kwargs["for_run"] == 7  # via Task-only SPECIAL
            assert "RUN_GROUPS" not in res

    def test_task_only_group_not_visible_to_other_tasks(self) -> None:
        """Test that if a group is defined for a Task only, other Tasks dont see it."""
        # fmt: off
        other_task_config: Dict[str, Any] = {
            "param1": 1,
        }
        # fmt: off

        with patch("lute.io.config.read_latest_db_entry") as mock_db:
            res: Dict[str, Any] = resolve_run_group_directives(
                "OtherTask", other_task_config, {}, "/tmp", 100
            )
            assert res["param1"] == 1
            mock_db.assert_not_called()

    def test_task_default_overrides_global_default(self) -> None:
        """Test the Task default trumps global."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "DARK": {
                "applies_to": {
                    "range": [0, 40],
                },
                "default": "@db:current",
            }
        }
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "DARK": {
                    "default": "@literal",
                },
            },
            "param1": 1,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry") as mock_db:
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", task_config, global_groups, "/tmp", 20
            )
            assert res["param1"] == 1

            mock_db.assert_not_called()

    def test_list_beats_overlapping_range(self) -> None:
        """Test a run group defined by list trumps a run group defined by range."""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "ALL": {
                "applies_to": {
                    "range": [0, 100],
                },
                "default": "@db:current",
            },
            "SPECIAL": {
                "applies_to": {
                    "runs": [50],
                },
                "default": "@literal",
            },
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=99) as mock_db:
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", {"param1": 1}, global_groups, "/tmp", 50
            )
            assert res["param1"] == 1  # SPECIAL (list) wins
            mock_db.assert_not_called()

            res = resolve_run_group_directives(
                "MyTask", {"param1": 1}, global_groups, "/tmp", 51
            )
            assert res["param1"] == 99  # Only ALL applies

    def test_narrower_range_beats_wider_range(self) -> None:
        """Test that narrower ranges trump broader ranges.

        NOTE: This should hold regardless of the order that the groups are defined in.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "WIDE": {
                "applies_to": {
                    "range": [0, 100],
                },
                "default": "@db:current",
            },
            "NARROW": {
                "applies_to": {
                    "range": [40, 60],
                },
                "default": "@literal",
            },
        }
        # fmt: on

        reversed_groups: Dict[str, Any] = dict(reversed(list(global_groups.items())))
        with patch("lute.io.config.read_latest_db_entry", return_value=99):
            for groups in (global_groups, reversed_groups):
                res: Dict[str, Any] = resolve_run_group_directives(
                    "MyTask", {"param1": 1}, groups, "/tmp", 50
                )
                assert res["param1"] == 1  # NARROW wins

                res = resolve_run_group_directives(
                    "MyTask", {"param1": 1}, groups, "/tmp", 70
                )
                assert res["param1"] == 99  # Only WIDE applies

    def test_specificity_beats_task_level_definition(self) -> None:
        """Ensure that a broader Task group doesn't override the narrower global group.

        NOTE: This is the only exception to the "Task trumps all"
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "NARROW": {
                "applies_to": {
                    "range": [45, 55],
                },
                "default": "@literal",
            },
        }
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "WIDE": {
                    "applies_to": {
                        "range": [0, 100],
                    },
                    "default": "@db:current",
                },
            },
            "param1": 1,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=99) as mock_db:
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", task_config, global_groups, "/tmp", 50
            )
            assert res["param1"] == 1
            mock_db.assert_not_called()

    def test_task_range_override_changes_specificity(self) -> None:
        """Test that in general, the Task overrides a global applies_to"""
        # fmt: off
        global_groups: Dict[str, Any] = {
            "A": {
                "applies_to": {
                    "range": [0, 100],
                },
                "default": "@db:current",
            },
            "B": {
                "applies_to": {
                    "range": [40, 60],
                },
                "default": "@literal",
            },
        }
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "A": {
                    "applies_to": {
                        "runs": [50],
                    },
                },
            },
            "param1": 1,
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=99):
            res: Dict[str, Any] = resolve_run_group_directives(
                "MyTask", copy.deepcopy(task_config), global_groups, "/tmp", 50
            )
            assert res["param1"] == 99  # A is now a list match -> beats B

            # Run 10 was in A's global range, but the Task override removed it.
            res = resolve_run_group_directives(
                "MyTask", copy.deepcopy(task_config), global_groups, "/tmp", 10
            )
            assert res["param1"] == 1

    def test_equal_specificity_warns(self) -> None:
        """Test that we warn if there are ties.

        NOTE: Ties fallback on the order things are defined in.
        """
        # fmt: off
        global_groups: Dict[str, Any] = {
            "FIRST": {
                "applies_to": {
                    "range": [0, 50],
                },
                "default": "@db:current",
            },
            "SECOND": {
                "applies_to": {
                    "range": [25, 75],
                },
                "default": "@literal",
            },
        }
        # fmt: on

        with patch("lute.io.config.read_latest_db_entry", return_value=99):
            with pytest.warns(UserWarning, match="equal specificity"):
                res: Dict[str, Any] = resolve_run_group_directives(
                    "MyTask", {"param1": 1}, global_groups, "/tmp", 30
                )
            assert res["param1"] == 99  # FIRST was defined first

    def test_run_groups_stripped_without_run(self) -> None:
        """Test that we don't leak the RUN_GROUPS special keys to pydantic."""
        # fmt: off
        task_config: Dict[str, Any] = {
            "RUN_GROUPS": {
                "A": {
                    "applies_to": {
                        "runs": [1],
                    },
                },
            },
            "param1": 1,
        }
        # fmt: on

        res: Dict[str, Any] = resolve_run_group_directives(
            "MyTask", task_config, {}, "/tmp", None
        )
        assert "RUN_GROUPS" not in res


# fmt: off
def _smd_config() -> Dict[str, Any]:
    return {
        "producer": "/path/to/producer.py",
        "producer_parameters": {
            "detnames": ["epix10k2M"],
            "getROIs": {
                "epix10k2M": [
                    {
                        "ROI": [[0, 10], [0, 10]],
                    },
                ],
            },
            "getAzIntParams": {
                "epix10k2M": {
                    "eBeam": 9.5,
                },
            },
        },
    }


_DB_PRODUCER_PARAMS: Dict[str, Any] = {
    "detnames": ["epix10k2M"],
    "getROIs": {
        "epix10k2M": [
            {
                "ROI": [[5, 50], [5, 50]],
            },
        ],
        "jungfrau1M": [
            {
                "ROI": [[1, 2], [3, 4]],
            },
        ],
    },
    "getAzIntParams": {
        "epix10k2M": {
            "eBeam": 18.0,
        },
    },
}
# fmt: on


def _mock_db(
    db_dir: str, task_name: str, param: str, for_run: int
) -> Optional[Dict[str, Any]]:
    if param == "producer_parameters":
        return copy.deepcopy(_DB_PRODUCER_PARAMS)
    return None


def _groups(policy: Dict[str, Any]) -> Dict[str, Any]:
    return {"ALL": {"applies_to": {"range": [0, 9999]}, **policy}}


class TestRunGroupNestedPolicies:
    def test_whole_top_level_dict_from_db(self):
        """A top-level policy replaces the entire nested dict."""
        with patch("lute.io.config.read_latest_db_entry", side_effect=_mock_db):
            res: Dict[str, Any] = resolve_run_group_directives(
                "SubmitSMD",
                _smd_config(),
                _groups({"producer_parameters": "@db:current"}),
                "/tmp",
                10,
            )

        assert res["producer_parameters"] == _DB_PRODUCER_PARAMS

    def test_nested_dict_from_db(self):
        """A dotted path replaces only that nested entry."""
        with patch("lute.io.config.read_latest_db_entry", side_effect=_mock_db) as m:
            res: Dict[str, Any] = resolve_run_group_directives(
                "SubmitSMD",
                _smd_config(),
                _groups({"producer_parameters.getROIs": "@db:current"}),
                "/tmp",
                10,
            )
            m.assert_called_once_with(
                db_dir="/tmp",
                task_name="SubmitSMD",
                param="producer_parameters",
                for_run=10,
            )

        params: Dict[str, Any] = res["producer_parameters"]
        assert params["getROIs"] == _DB_PRODUCER_PARAMS["getROIs"]
        assert params["getAzIntParams"] == {"epix10k2M": {"eBeam": 9.5}}  # untouched

    def test_single_nested_item_from_db(self):
        """Deeper paths work too, e.g. one detector's parameters."""
        with patch("lute.io.config.read_latest_db_entry", side_effect=_mock_db):
            res: Dict[str, Any] = resolve_run_group_directives(
                "SubmitSMD",
                _smd_config(),
                _groups({"producer_parameters.getAzIntParams.epix10k2M": "@db:run:5"}),
                "/tmp",
                10,
            )

        params: Dict[str, Any] = res["producer_parameters"]
        assert params["getAzIntParams"]["epix10k2M"] == {"eBeam": 18.0}
        assert params["getROIs"] == {"epix10k2M": [{"ROI": [[0, 10], [0, 10]]}]}
