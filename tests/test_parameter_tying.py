"""Tests for tied-parameter detection.

Covers:
- a shared parameter is actually found (Paddle de-duplicates by default, which
  hid every tie)
- several names for one parameter are all reported
- an untied model is still reported as untied
"""

import paddle
import pytest

from ocean.utils.parameter_tying import assert_no_tying_parameters, find_tying_parameters


def test_no_tying_in_a_plain_model():
    net = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.Linear(4, 4))
    assert find_tying_parameters(net) == []
    assert_no_tying_parameters(net)  # does not raise


def test_a_shared_parameter_is_found():
    """named_parameters() de-duplicates by default, so the shared parameter was
    listed once and the duplicate could never be seen."""
    net = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.Linear(4, 4))
    net[1].weight = net[0].weight

    assert find_tying_parameters(net) == [("0.weight", "1.weight")]


def test_assert_raises_on_a_shared_parameter():
    net = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.Linear(4, 4))
    net[1].weight = net[0].weight

    with pytest.raises(ValueError, match="tied parameters"):
        assert_no_tying_parameters(net)


def test_three_way_tie_reports_every_extra_name():
    net = paddle.nn.Sequential(paddle.nn.Linear(4, 4), paddle.nn.Linear(4, 4), paddle.nn.Linear(4, 4))
    net[1].weight = net[0].weight
    net[2].weight = net[0].weight

    assert find_tying_parameters(net) == [("0.weight", "1.weight"), ("0.weight", "2.weight")]


def test_tying_inside_a_nested_model():
    class Nested(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.encoder = paddle.nn.Linear(4, 4)
            self.decoder = paddle.nn.Linear(4, 4)
            self.decoder.weight = self.encoder.weight

    tied = find_tying_parameters(Nested())
    assert tied == [("encoder.weight", "decoder.weight")]
