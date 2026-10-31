"""HandTracker: stable IDs, bystander filter, stable handedness, and smoothing."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import replace

from ai_playground.config import TrackingConfig
from ai_playground.gestures.features import WRIST, hand_scale_image
from ai_playground.tracking.filters import OneEuroFilter
from ai_playground.types import Hand, Handedness, HandFrame, TrackedHand


class _Track:
    def __init__(self, tid: int, cfg: TrackingConfig) -> None:
        self.id = tid
        self.age = 0
        self.missed = 0
        self.votes: deque[Handedness] = deque(maxlen=cfg.handedness_window)
        self.f_image = OneEuroFilter(cfg.one_euro.image)
        self.f_world = OneEuroFilter(cfg.one_euro.world)
        self.hand: Hand | None = None

    @property
    def wrist(self) -> tuple[float, float]:
        assert self.hand is not None
        return float(self.hand.landmarks[WRIST, 0]), float(self.hand.landmarks[WRIST, 1])

    def observe(self, det: Hand, t: float) -> None:
        self.votes.append(det.handedness)
        # Majority vote; ties go to the newest label.
        counts = Counter(self.votes)
        best = max(counts.values())
        handedness = next(h for h in reversed(self.votes) if counts[h] == best)
        self.hand = replace(
            det,
            handedness=handedness,
            landmarks=self.f_image(det.landmarks, t),
            world=self.f_world(det.world, t),
        )
        self.missed = 0
        self.age += 1


class HandTracker:
    def __init__(self, cfg: TrackingConfig) -> None:
        self.cfg = cfg
        self._tracks: list[_Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, frame: HandFrame) -> list[TrackedHand]:
        cfg = self.cfg
        dets = [
            h for h in frame.hands if hand_scale_image(h.landmarks, frame.size) >= cfg.min_hand_size
        ]

        # Greedy nearest-wrist matching; with at most 2 hands this is as good as Hungarian.
        pairs = []
        for ti, tr in enumerate(self._tracks):
            tx, ty = tr.wrist
            for di, d in enumerate(dets):
                dist = math.hypot(
                    float(d.landmarks[WRIST, 0]) - tx, float(d.landmarks[WRIST, 1]) - ty
                )
                if dist <= cfg.match_gate:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self._tracks[ti].observe(dets[di], frame.t_capture)

        survivors = []
        for ti, tr in enumerate(self._tracks):
            if ti not in used_t:
                tr.missed += 1
                if tr.missed > cfg.grace_frames:
                    continue
            survivors.append(tr)
        for di, d in enumerate(dets):
            if di not in used_d:
                tr = _Track(self._next_id, cfg)
                self._next_id += 1
                tr.observe(d, frame.t_capture)
                survivors.append(tr)
        self._tracks = survivors

        return [
            TrackedHand(id=tr.id, hand=tr.hand, age=tr.age, missed=tr.missed)
            for tr in sorted(self._tracks, key=lambda tr: tr.id)
            if tr.hand is not None
        ]
