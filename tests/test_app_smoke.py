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
