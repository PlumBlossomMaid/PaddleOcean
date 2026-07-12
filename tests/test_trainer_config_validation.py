"""D2: construction-time validation of validation-schedule config.

These parameters are validated in ``_DataConnector.on_trainer_init`` (at Trainer
construction) so misconfiguration fails fast instead of silently mis-scheduling
validation once fit() runs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import ocean
from ocean.utils import MisconfigurationException


def _trainer(**kwargs):
    return ocean.Trainer(max_epochs=1, logger=False, enable_checkpointing=False, **kwargs)


def test_check_val_every_n_epoch_must_be_int():
    with pytest.raises(MisconfigurationException):
        _trainer(check_val_every_n_epoch=1.5)


def test_float_val_check_interval_requires_epoch_cadence():
    """check_val_every_n_epoch=None forbids a fractional val_check_interval."""
    with pytest.raises(MisconfigurationException):
        _trainer(check_val_every_n_epoch=None, val_check_interval=0.5)


def test_reload_dataloaders_must_be_non_negative_int():
    with pytest.raises(MisconfigurationException):
        _trainer(reload_dataloaders_every_n_epochs=-1)


def test_none_epoch_cadence_with_int_interval_is_valid():
    # int interval may span epochs when epoch gating is disabled
    _trainer(check_val_every_n_epoch=None, val_check_interval=2)


def test_defaults_are_valid():
    _trainer()


def test_int_epoch_cadence_is_valid():
    _trainer(check_val_every_n_epoch=2)


def test_time_based_interval_with_none_cadence_is_valid():
    # time-based interval is allowed even without epoch gating
    _trainer(check_val_every_n_epoch=None, val_check_interval="00:00:01:00")
