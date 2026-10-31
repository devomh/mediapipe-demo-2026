import pytest

from ai_playground import config


def test_defaults_load(cfg):
    assert cfg.camera.width == 640
    assert cfg.gestures.pinch.enter < cfg.gestures.pinch.exit
    assert cfg.model_path().name == "gesture_recognizer.task"


def test_override_file_deep_merges(tmp_path):
    p = tmp_path / "venue.toml"
    p.write_text("[camera]\nindex = 2\n[tracking.one_euro.image]\nbeta = 3\n")
    c = config.load(p)
    assert c.camera.index == 2
    assert c.camera.width == 640  # untouched sibling keeps its default
    assert c.tracking.one_euro.image.beta == 3.0  # int coerced to float
    assert c.tracking.one_euro.world.beta == config.load().tracking.one_euro.world.beta


def test_unknown_key_fails_fast(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[camera]\nindx = 2\n")
    with pytest.raises(config.ConfigError, match="indx"):
        config.load(p)


def test_unknown_section_fails_fast():
    with pytest.raises(config.ConfigError, match="audoi"):
        config.load(overrides={"audoi": {}})


@pytest.mark.parametrize(
    "override",
    [
        {"camera": {"index": "0"}},
        {"camera": {"index": 1.5}},
        {"display": {"fullscreen": 1}},
        {"gestures": {"pinch": {"enter": "low"}}},
    ],
)
def test_wrong_types_fail(override):
    with pytest.raises(config.ConfigError):
        config.load(overrides=override)


def test_frozen(cfg):
    with pytest.raises(AttributeError):
        cfg.camera.index = 3  # type: ignore[misc]
