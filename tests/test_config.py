"""Tests for stage configs."""
import pytest

from config import Config


def test_stage_64():
    cfg = Config.for_stage(64)
    assert cfg.image_size == 64
    assert cfg.base_channels == 64
    assert cfg.run_name == "stage-64"


def test_stage_256_is_full_spec():
    cfg = Config.for_stage(256)
    assert cfg.channel_mults == (1, 2, 4, 8)
    assert cfg.base_channels == 128


def test_overrides_win():
    cfg = Config.for_stage(64, batch_size=2, run_name="smoke")
    assert cfg.batch_size == 2
    assert cfg.run_name == "smoke"


def test_unknown_stage_raises():
    with pytest.raises(ValueError):
        Config.for_stage(512)


def test_to_dict_round_trips():
    cfg = Config.for_stage(64)
    clone = Config(**cfg.to_dict())
    assert clone.image_size == cfg.image_size
    assert clone.channel_mults == cfg.channel_mults
