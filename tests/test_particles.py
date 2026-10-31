"""Particle pool and glow cache (Phase A §4)."""

import numpy as np
import pygame
import pytest

from ai_playground.render.particles import GlowCache, Particles


@pytest.fixture
def screen():
    pygame.init()
    yield pygame.display.set_mode((640, 480))
    pygame.quit()


def test_emit_update_and_expire():
    p = Particles(100, seed=0)
    p.emit(10, (100, 100), hue=2, speed=(100, 100), life=(0.5, 0.5))
    assert p.alive == 10
    p.update(0.25, gravity=0.0, drag=0.0)
    dist = np.hypot(*(p.pos[p.life > 0] - (100, 100)).T)
    assert dist == pytest.approx(np.full(10, 25.0), abs=0.01)  # 100 px/s for 0.25 s
    p.update(0.3)
    assert p.alive == 0


def test_gravity_pulls_down():
    p = Particles(10, seed=0)
    p.emit(5, (0, 0), hue=0, speed=(0, 0), life=(2, 2))
    p.update(0.5, gravity=1000.0, drag=0.0)
    assert (p.pos[p.life > 0, 1] > 0).all() and np.allclose(p.vel[p.life > 0, 1], 500)


def test_full_pool_reuses_the_particles_nearest_death():
    p = Particles(20, seed=0)
    p.emit(10, (0, 0), hue=0, life=(0.1, 0.1))  # short-lived
    p.emit(10, (0, 0), hue=1, life=(5, 5))  # long-lived
    p.emit(5, (50, 50), hue=3, life=(1, 1))  # pool full: replace 5 short-lived ones
    assert p.alive == 20
    assert np.count_nonzero(p.hue == 1) == 10  # long-lived ones untouched
    assert np.count_nonzero(p.hue == 3) == 5


def test_emit_more_than_the_cap():
    p = Particles(8, seed=0)
    p.emit(50, (0, 0), hue=0)
    assert p.alive == 8


def test_draw_full_pool(screen):
    p = Particles(1500, seed=0)
    for i in range(50):
        p.emit(30, (20 + 12 * i, 240), hue=i % 8)
    screen.fill((0, 0, 0))
    p.draw(screen)
    assert screen.get_at((20 + 12 * 25, 240))[:3] != (0, 0, 0)  # additive light landed


def test_glow_sizes_are_bucketed_and_scaled(screen):
    g = GlowCache(scale=1.7, bucket_px=8)
    a, b = g.sprite(0, 50), g.sprite(0, 51)
    assert a is b  # same bucket: cached
    big = g.sprite(0, 300)
    assert big.get_width() == 2 * round(300 * 1.7 / 8) * 8 + 1
    centre = big.get_at((big.get_width() // 2, big.get_width() // 2))
    edge = big.get_at((0, big.get_width() // 2))
    assert sum(centre[:3]) > 0 and sum(edge[:3]) == 0  # soft falloff to black


def test_glow_fades_with_alpha(screen):
    g = GlowCache()
    bright, faint = g.sprite(4, 40, alpha=1.0), g.sprite(4, 40, alpha=0.2)
    c = bright.get_width() // 2
    assert sum(faint.get_at((c, c))[:3]) < sum(bright.get_at((c, c))[:3])
    g.draw(screen, (100, 100), 4, 40, alpha=0.0)  # fully faded: nothing drawn, no error
