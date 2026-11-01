"""Smoke: the Vision scene runs on replay with SDL_VIDEODRIVER=dummy (set in conftest)."""

import pygame
import pytest
from synthetic import frame, make_hand

from ai_playground.__main__ import main
from ai_playground.app import App, RunOptions
from ai_playground.sources.recorder import Recorder
from ai_playground.sources.replay import ReplaySource, load_clip


@pytest.fixture
def clip(tmp_path):
    p = tmp_path / "smoke.jsonl.gz"
    rec = Recorder(p, note="smoke")
    for i in range(60):
        hands = [
            make_hand(pinch=0.1 if 20 <= i < 40 else 1.0, center=(0.3 + 0.005 * i, 0.5)),
            make_hand(
                handedness="Left",
                fist=True,
                center=(0.25, 0.55),
                gesture="Closed_Fist",
                gesture_score=0.9,
            ),
        ]
        rec.write(frame(i + 1, i / 30, hands))
    rec.close()
    return p


def test_vision_scene_runs_120_frames_on_replay(cfg, clip, tmp_path):
    pygame.init()
    try:
        out = tmp_path / "rerecorded.jsonl"
        app = App(
            cfg,
            ReplaySource(clip, loop=True, speed=4.0),
            RunOptions(source_label="REPLAY", max_frames=120, record_path=out),
        )
        app.overlay.show_indices = True
        assert app.run() == 0
        assert app.frames_rendered == 120
        assert app.hands_tracked == 2
    finally:
        pygame.quit()
    assert len(load_clip(out).frames) > 10  # recording from replay works too


def test_hotkeys_do_not_crash(cfg, clip, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the R hotkey records into ./recordings/
    pygame.init()
    try:
        app = App(cfg, ReplaySource(clip, loop=True), RunOptions(max_frames=5))
        for key in (
            pygame.K_v,
            pygame.K_v,
            pygame.K_h,
            pygame.K_b,
            pygame.K_b,
            pygame.K_b,
            pygame.K_l,
            pygame.K_r,
            pygame.K_SPACE,
            pygame.K_r,
        ):
            app._on_key(key)
        assert app.recorder is None
        assert len(list((tmp_path / "recordings").glob("*.jsonl.gz"))) == 1
        assert app.run() == 0
    finally:
        pygame.quit()


def test_cli_replay(clip):
    assert main(["--replay", str(clip), "--max-frames", "30"]) == 0


def test_cli_rejects_unknown_scene(clip):
    assert main(["--replay", str(clip), "--scene", "nope"]) == 2


def test_keys_overlay_toggles_and_lists_only_usable_keys(cfg, clip):
    pygame.init()
    try:
        app = App(cfg, ReplaySource(clip), RunOptions(scene="vision", max_frames=1))
        assert app.keys_overlay.visible == cfg.hud.keys_visible
        app._on_key(pygame.K_k)
        assert app.keys_overlay.visible != cfg.hud.keys_visible

        def listed():
            return [h.label for h in app.hotkeys if h.is_available()]

        assert "C" not in listed() and "Space" not in listed()  # Vision: no shapes, no recording
        app._on_key(pygame.K_2)
        assert app.scene.name == "geometry" and "C" in listed()
        assert not app.overlay.visible  # Geometry starts with the Vision overlay hidden
        assert "L" not in listed()  # landmark numbers only make sense with the overlay on
        app._on_key(pygame.K_SPACE)  # not recording: ignored, no crash
        app.keys_overlay.visible = True
        app.keys_overlay.draw(app.surface, app.hotkeys)
    finally:
        pygame.quit()


def test_effects_key_toggles_geometry_effects_and_is_hidden_in_vision(cfg, clip):
    pygame.init()
    try:
        app = App(cfg, ReplaySource(clip), RunOptions(scene="vision", max_frames=1))
        assert "E" not in [h.label for h in app.hotkeys if h.is_available()]
        app._on_key(pygame.K_e)  # no effect, no crash in Vision
        app._on_key(pygame.K_2)
        assert app.scene.effects_on is True
        app._on_key(pygame.K_e)
        assert app.scene.effects_on is False
        hk = next(h for h in app.hotkeys if h.label == "E")
        assert hk.is_available() and hk.state() == "off"
        app._on_key(pygame.K_1)
        app._on_key(pygame.K_2)  # a fresh Geometry scene starts from the config default
        assert app.scene.effects_on is cfg.effects.enabled
    finally:
        pygame.quit()


def test_m_mutes_sound_and_is_hidden_without_it(cfg, clip):
    from ai_playground import config

    pygame.init()
    try:
        app = App(cfg, ReplaySource(clip), RunOptions(scene="geometry", max_frames=1))
        assert app.sfx.available and app.ctx.sfx is app.sfx  # dummy audio driver
        hk = next(h for h in app.hotkeys if h.label == "M")
        assert hk.is_available() and hk.state() == "on"
        app._on_key(pygame.K_m)
        assert app.sfx.muted and hk.state() == "off"
        assert app.run() == 0
    finally:
        pygame.quit()
    pygame.init()
    try:
        quiet = config.load(overrides={"sound": {"enabled": False}})
        app = App(quiet, ReplaySource(clip), RunOptions(scene="geometry", max_frames=1))
        assert not app.sfx.available
        assert "M" not in [h.label for h in app.hotkeys if h.is_available()]
        app._on_key(pygame.K_m)  # no crash
    finally:
        pygame.quit()
