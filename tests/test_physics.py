"""Physics world, headless (Phase A §8)."""

import math

import pytest

from ai_playground.physics.world import KINDS, World, shape_vertices

SIZE = (1280, 720)


@pytest.fixture
def world(cfg):
    return World(SIZE, cfg.physics)


def run(world, seconds, fps=60):
    for _ in range(round(seconds * fps)):
        world.step(1 / fps)


@pytest.mark.parametrize("kind", KINDS)
def test_spawned_shape_falls_and_lands_on_the_floor(world, kind):
    p = world.spawn(kind, (640, 100), 40)
    run(world, 0.2)
    assert p.body.position.y > 100  # falling
    run(world, 3.0)
    bottom = max(y for _, y in _extent(p))
    assert SIZE[1] - 8 < bottom < SIZE[1] + 2  # resting on the floor


def _extent(p):
    if p.kind == "circle":
        x, y = p.body.position
        return [(x, y + p.radius)]
    return [tuple(p.body.local_to_world(v)) for v in p.shape.get_vertices()]


def test_gravity_scales_with_window_height(cfg):
    small, big = World((640, 360), cfg.physics), World((1920, 1080), cfg.physics)
    assert big.space.gravity.y == pytest.approx(3 * small.space.gravity.y)
    assert big.space.gravity.y == pytest.approx(cfg.physics.gravity_factor * 1080)


def test_release_velocity_is_kept_and_capped(world, cfg):
    p = world.spawn("square", (640, 300), 30, vel=(800, -300))
    assert tuple(p.body.velocity) == (800, -300)
    fast = world.spawn("circle", (640, 300), 30, vel=(30000, 0))
    assert math.hypot(*fast.body.velocity) == pytest.approx(cfg.physics.max_speed * SIZE[1])


def test_thrown_shapes_stay_between_the_walls(world):
    left = world.spawn("circle", (300, 300), 30, vel=(-3000, -1500))
    right = world.spawn("pentagon", (900, 300), 30, vel=(3000, -2500))
    run(world, 4.0)
    for p in (left, right):
        assert 0 < p.body.position.x < SIZE[0]
        assert p.body.position.y < SIZE[1]
    assert len(world.pieces) == 2


def test_hard_upward_throw_rises_one_window_height_and_comes_back(world, cfg):
    """Open ceiling: a shape may leave the top of the screen, but only briefly."""
    p = world.spawn("triangle", (640, 300), 30, vel=(400, -6000))
    top = 300.0
    for _ in range(round(2.5 * 60)):
        world.step(1 / 60)
        top = min(top, p.body.position.y)
    assert top < 0  # it did leave the screen
    assert 300 - top == pytest.approx(cfg.physics.max_rise * SIZE[1], rel=0.05)
    assert 0 < p.body.position.y < SIZE[1]  # back on screen within 2.5 s


def test_body_cap_fades_out_the_oldest(world, cfg):
    cap = cfg.physics.max_bodies
    first = world.spawn("circle", (100, 100), 10)
    for i in range(cap):
        world.spawn("circle", (100 + 25 * (i % 40), 200), 10)
    assert first.fade == 0.0 and first.alpha == 1.0
    assert sum(p.fade is None for p in world.pieces) == cap
    run(world, cfg.physics.fade_s / 2)
    assert 0.0 < first.alpha < 1.0
    run(world, cfg.physics.fade_s)
    assert first not in world.pieces
    assert len(world.pieces) == cap


def test_substeps_are_capped(world, cfg):
    assert world.step(1 / 60) == 2  # 120 Hz physics at 60 FPS
    assert world.step(1.0) == cfg.physics.max_substeps  # a long stall drops time


def test_impacts_are_recorded(world):
    world.spawn("square", (640, 600), 30, vel=(0, 2000))
    hits = []
    for _ in range(60):
        world.step(1 / 60)
        hits += world.impacts
    assert hits
    assert max(h.impulse for h in hits) > 0
    assert any(None in h.pieces for h in hits)  # the floor is a wall, not a piece


def test_resize_moves_the_floor(world):
    world.resize((800, 400))
    p = world.spawn("circle", (400, 100), 20)
    run(world, 3.0)
    assert p.body.position.y == pytest.approx(400 - 20, abs=3)


def test_clear(world):
    for i in range(5):
        world.spawn("bar", (200 + 100 * i, 100), 40)
    world.clear()
    assert world.pieces == [] and len(world.space.bodies) == 0


@pytest.mark.parametrize("kind,n", [("bar", 4), ("triangle", 3), ("square", 4), ("pentagon", 5)])
def test_shape_vertices(kind, n):
    v = shape_vertices(kind, 40)
    assert len(v) == n
    if kind == "bar":
        xs, ys = [x for x, _ in v], [y for _, y in v]
        assert (max(xs) - min(xs)) / (max(ys) - min(ys)) == pytest.approx(4.0)
    else:
        assert all(math.hypot(x, y) == pytest.approx(40) for x, y in v)
    if kind == "square":
        assert sorted({round(y, 6) for _, y in v}) == pytest.approx([-28.284271, 28.284271])
