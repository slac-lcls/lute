import pytest

from maestro.parser_params import (
    expand_param_matrix,
    zip_param_matrix,
    validate_param_matrix,
)


class TestExpandParamMatrix:
    def test_product(self):
        param_matrix = {"num_cores": [1, 2], "batch_size": [100, 500]}
        combos = expand_param_matrix(param_matrix)
        assert len(combos) == 4
        assert {"num_cores": 1, "batch_size": 100} in combos
        assert {"num_cores": 2, "batch_size": 500} in combos

    def test_empty(self):
        assert expand_param_matrix({}) == [{}]


class TestZipParamMatrix:
    def test_zip(self):
        param_matrix = {
            "detname": ["Quad0", "Quad1"],
            "dist": [0.28, 0.19],
        }
        combos = zip_param_matrix(param_matrix)
        assert combos == [
            {"detname": "Quad0", "dist": 0.28},
            {"detname": "Quad1", "dist": 0.19},
        ]

    def test_empty(self):
        assert zip_param_matrix({}) == [{}]

    def test_mismatched_lengths_raises(self):
        param_matrix = {"detname": ["Quad0", "Quad1"], "dist": [0.28]}
        with pytest.raises(ValueError):
            zip_param_matrix(param_matrix)


class TestValidateParamMatrix:
    def test_valid(self):
        validate_param_matrix({"num_cores": [1, 2]})

    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError):
            validate_param_matrix([1, 2])

    def test_non_list_value_raises(self):
        with pytest.raises(ValueError):
            validate_param_matrix({"num_cores": 1})

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            validate_param_matrix({"num_cores": []})

    def test_require_equal_length_ok(self):
        validate_param_matrix(
            {"detname": ["Quad0", "Quad1"], "dist": [0.28, 0.19]},
            require_equal_length=True,
        )

    def test_require_equal_length_raises(self):
        with pytest.raises(ValueError):
            validate_param_matrix(
                {"detname": ["Quad0", "Quad1"], "dist": [0.28]},
                require_equal_length=True,
            )
