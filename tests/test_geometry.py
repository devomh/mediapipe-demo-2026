"""Magic Geometry scene: gesture events -> shapes. Driven directly, no threads."""

from pathlib import Path

import pygame
import pytest

from ai_playground.gestures.engine import GestureEngine
from ai_playground.render.background import Background
from ai_playground.render.viewport import Viewport
from ai_playground.scenes.base import AppContext
from ai_playground.scenes.geometry import GeometryScene
from ai_playground.sources.replay import load_clip
from ai_playground.tracking.identity import HandTracker
from ai_playground.types import GestureEvent

FIXTURES = Path(__file__).parent / "fixtures"
WINDOW = (1280, 720)


@pytest.fixture
def scene(cfg):
    pygame.init()
    surface = pygame.display.set_mode(WINDOW)
    ctx = AppContext(
        config=cfg,
        viewport=Viewport(WINDOW, (640, 480)),
        background=Background(),
        switch_scene=lambda name: None,
    )
    s = GeometryScene(seed=1)
    s.enter(ctx)
    s.surface = surface
    yield s
    pygame.quit()


def ev(type_, t, pos=(0.5, 0.5), vel=(0.0, 0.0), hand=1, **data):
    return GestureEvent(type_, hand, "Right", t, pos, vel=vel, data=data)


def test_pinch_hold_release_spawns_an_inflated_shape(scene, cfg):
    scene.handle([ev("pinch_start", 0.0)], [])
    assert len(scene.pending) == 1 and scene.world.pieces == []
    r0 = scene.pending[1].radius
    scene.handle([ev("pinch_move", cfg.geometry.inflate_s / 2, pos=(0.6, 0.4))], [])
    p = scene.pending[1]
    assert r0 < p.radius < cfg.geometry.max_size * WINDOW[1]
    assert p.pos == scene.ctx.viewport.to_px(0.6, 0.4)  # follows the pinch
    scene.handle([ev("pinch_move", 10.0)], [])
    assert scene.pending[1].radius == pytest.approx(cfg.geometry.max_size * WINDOW[1])
    scene.handle([ev("pinch_end", 10.1, reason="released")], [])
    assert scene.pending == {}
    (piece,) = scene.world.pieces
    assert piece.radius == pytest.approx(cfg.geometry.max_size * WINDOW[1])


def test_release_velocity_becomes_a_throw(scene):
    vp = scene.ctx.viewport
    scene.handle(
        [ev("pinch_start", 0.0), ev("pinch_end", 0.5, vel=(0.5, -0.25), reason="released")], []
    )
    (piece,) = scene.world.pieces
    assert tuple(piece.body.velocity) == pytest.approx(
        (0.5 * vp.dest_size[0], -0.25 * vp.dest_size[1])
    )


def test_lost_hand_drops_without_throwing(scene):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.5, vel=(2.0, -2.0), reason="lost")], [])
    (piece,) = scene.world.pieces
    assert tuple(piece.body.velocity) == (0.0, 0.0)


def test_kinds_cycle_and_hues_differ(scene):
    for i in range(6):
        scene.handle([ev("pinch_start", i), ev("pinch_end", i + 0.2, reason="released")], [])
    kinds = [p.kind for p in scene.world.pieces]
    assert kinds == ["circle", "bar", "triangle", "square", "pentagon", "circle"]
    hues = [p.hue for p in scene.world.pieces]
    assert all(a != b for a, b in zip(hues, hues[1:], strict=False))


def test_two_hands_make_shapes_independently(scene):
    scene.handle(
        [ev("pinch_start", 0.0, hand=1), ev("pinch_start", 0.1, hand=2, pos=(0.2, 0.2))], []
    )
    assert set(scene.pending) == {1, 2}
    scene.handle([ev("pinch_end", 0.5, hand=2, reason="released")], [])
    assert set(scene.pending) == {1} and len(scene.world.pieces) == 1


def test_update_and_draw_run(scene):
    scene.handle([ev("pinch_start", 0.0)], [])
    for _ in range(30):
        scene.update(1 / 60)
        scene.draw(scene.surface)
    scene.handle([ev("pinch_end", 0.6, reason="released")], [])
    y0 = scene.world.pieces[0].body.position.y
    for _ in range(30):
        scene.update(1 / 60)
        scene.draw(scene.surface)
    assert scene.world.pieces[0].body.position.y > y0


def test_clear(scene):
    scene.handle([ev("pinch_start", 0.0), ev("pinch_end", 0.2, reason="released")], [])
    scene.handle([ev("pinch_start", 0.3)], [])
    scene.clear()
    assert scene.world.pieces == [] and scene.pending == {}


@pytest.mark.parametrize("name,shapes", [("right_pinch_x5", 5), ("right_pinch_hold", 1)])
def test_recorded_pinches_make_shapes(scene, cfg, name, shapes):
    """Real hands: each recorded pinch becomes exactly one shape."""
    path = FIXTURES / f"{name}.jsonl.gz"
    if not path.exists():
        pytest.skip(f"fixture {name} not recorded")
    clip = load_clip(path)
    tracker = HandTracker(cfg.tracking)
    engine = GestureEngine(cfg.gestures, tuple(clip.header["size"]))
    prev = clip.frames[0].t_capture
    for fr in clip.frames:
        events, states = engine.update(tracker.update(fr), fr.t_capture)
        scene.handle(events, states)
        scene.update(fr.t_capture - prev)
        prev = fr.t_capture
    assert len(scene.world.pieces) == shapes
    assert scene.pending == {}
