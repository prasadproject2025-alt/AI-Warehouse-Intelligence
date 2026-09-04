"""
Behaviour detectors: positive cases, and — more importantly — the negative
cases that used to produce false incidents.

Tracks are scripted synthetically so these tests verify the *reasoning*, not
the object detector.
"""

import pytest

from behaviour.base import BehaviourType
from behaviour.kinematic_detectors import DragDetector, DropDetector, RollDetector, ThrowDetector
from behaviour.scene_detectors import DockDetector, WetFloorDetector
from behaviour.spatial_detectors import (
    DesignatedAreaDetector,
    OrientationDetector,
    StackingDetector,
    SteppingDetector,
)
from detection.object_classes import WarehouseEntity
from detection.tracker import MotionState
from helpers import FRAME_H, base_context, build_track

FPS = 10.0


def _set_states(track, states, contact=True):
    """Stamp motion states and operator contact onto a scripted history."""
    for h, s in zip(track.history, states):
        h["state"] = s.value
        h["operator_contact"] = 3 if contact else None
    track.state = states[-1]
    track.state_history = [(s.value, i / FPS) for i, s in enumerate(states)]


# ------------------------------------------------------------------- drop
def test_drop_fires_on_full_temporal_chain():
    """carried -> sustained descent -> impact -> at rest must produce one event."""
    path = (
        [(600, 200)] * 4                              # held steady
        + [(600, 250), (600, 320), (600, 410), (600, 500)]  # falling
        + [(600, 520)] * 6                            # impact then at rest
    )
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.CARRIED] * 8 + [MotionState.SETTLED] * 6)

    events = DropDetector().process([trk], 14, 1.4, base_context())
    assert len(events) == 1
    ev = events[0]
    assert ev.behaviour_type is BehaviourType.PRODUCT_DROP
    assert ev.evidence_stages, "a drop must carry its temporal evidence"
    assert ev.risk_factors


def test_drop_does_not_fire_without_impact():
    """A steady descent that never stops is a controlled lowering, not a drop."""
    path = [(600, 200 + i * 22) for i in range(14)]  # still moving at the end
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.CARRIED] * 14)
    assert DropDetector().process([trk], 14, 1.4, base_context()) == []


def test_drop_does_not_fire_on_a_stationary_object():
    trk = build_track([(600, 400)] * 15, fps=FPS)
    _set_states(trk, [MotionState.STATIONARY] * 15, contact=False)
    assert DropDetector().process([trk], 15, 1.5, base_context()) == []


def test_drop_cooldown_prevents_duplicate_events():
    path = ([(600, 200)] * 4 + [(600, 250), (600, 320), (600, 410), (600, 500)]
            + [(600, 520)] * 6)
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.CARRIED] * 8 + [MotionState.SETTLED] * 6)
    det = DropDetector(cooldown_sec=4.0)
    first = det.process([trk], 14, 1.4, base_context())
    second = det.process([trk], 15, 1.5, base_context())
    assert len(first) == 1 and second == []


# ------------------------------------------------------------------ throw
def test_throw_requires_release_from_operator_contact():
    path = [(300, 300)] * 3 + [(360, 310), (440, 330), (530, 370), (620, 430), (700, 500)]
    trk = build_track(path, fps=FPS)
    states = [MotionState.CARRIED] * 3 + [MotionState.FALLING] * 5
    _set_states(trk, states)
    for h in trk.history[3:]:
        h["operator_contact"] = None  # released

    events = ThrowDetector().process([trk], 8, 0.8, base_context())
    assert len(events) == 1
    assert events[0].behaviour_type is BehaviourType.PRODUCT_THROW


def test_throw_ignores_object_never_in_operator_contact():
    """Fast motion with nobody handling it is not a throw."""
    path = [(300 + i * 90, 300 + i * 40) for i in range(8)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.FALLING] * 8, contact=False)
    assert ThrowDetector().process([trk], 8, 0.8, base_context()) == []


def test_pure_vertical_fall_is_not_classified_as_a_throw():
    path = [(600, 200)] * 3 + [(600, 280), (600, 380), (600, 480), (600, 560), (600, 570)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.CARRIED] * 3 + [MotionState.FALLING] * 5)
    for h in trk.history[3:]:
        h["operator_contact"] = None
    assert ThrowDetector().process([trk], 8, 0.8, base_context()) == []


# ------------------------------------------------------------------- drag
def test_drag_requires_sustained_sliding_with_an_operator():
    path = [(200 + i * 26, 620) for i in range(16)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 16)
    events = DragDetector().process([trk], 16, 1.6, base_context())
    assert len(events) == 1
    assert events[0].duration_sec >= 1.0


def test_drag_ignores_short_bursts():
    path = [(200, 620), (215, 620), (232, 620), (250, 620), (268, 620), (286, 620)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 6)
    assert DragDetector().process([trk], 6, 0.6, base_context()) == []


def test_drag_ignores_movement_with_no_operator_nearby():
    """A product on a moving conveyor or trolley is not being dragged by hand."""
    path = [(200 + i * 26, 620) for i in range(16)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 16, contact=False)
    assert DragDetector().process([trk], 16, 1.6, base_context()) == []


# ------------------------------------------------------------------- roll
def test_roll_requires_repeated_aspect_inversion_and_translation():
    path = []
    for i in range(14):
        wide = i % 2 == 0
        path.append((200 + i * 22, 640, 180 if wide else 90, 90 if wide else 180))
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 14)
    events = RollDetector().process([trk], 14, 1.4, base_context())
    assert len(events) == 1
    assert events[0].metadata["inversion_cycles"] >= 2


def test_roll_ignores_a_stable_carried_box():
    path = [(200 + i * 22, 640, 150, 120) for i in range(14)]
    trk = build_track(path, fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 14)
    assert RollDetector().process([trk], 14, 1.4, base_context()) == []


# --------------------------------------------------------------- stepping
def _ground_plane(residual_slope=-1.0):
    """Ground plane where taller (nearer) people have feet lower in frame."""
    def fn(person_height_norm):
        return 0.95 + residual_slope * (0.45 - person_height_norm) * 0.9
    return fn


def test_stepping_fires_when_operator_is_elevated_at_their_own_depth():
    # Operator height 0.45 -> expected floor y = 0.95; feet at 0.70 => elevated.
    op = build_track([(600, 720 * 0.70 - 162)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 324.0), fps=FPS, track_id=3)
    prod = build_track([(600, 720 * 0.76)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())

    det = SteppingDetector()
    # Confirmation needs MIN_OBSERVATIONS independent sightings, not just a
    # start and an end, so a single-frame coincidence cannot raise an incident.
    det.process([op, prod], 6, 0.0, ctx)
    det.process([op, prod], 9, 0.3, ctx)
    events = det.process([op, prod], 12, 0.6, ctx)
    assert len(events) == 1
    assert events[0].behaviour_type is BehaviourType.STEPPING_ON_CARTON


def test_stepping_does_not_fire_for_a_distant_operator_on_the_floor():
    """
    Regression: a small (distant) operator has feet high in the frame purely
    because of perspective. Against a fixed floor line this produced constant
    false positives; against the depth-aware plane it must produce none.
    """
    # Height 0.18 (far away) -> expected floor y = 0.71; feet exactly there.
    h_px = 720 * 0.18
    feet_y = 720 * 0.7157
    op = build_track([(600, feet_y - h_px / 2)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(50.0, h_px), fps=FPS, track_id=3)
    prod = build_track([(600, feet_y + 40)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())

    det = SteppingDetector()
    det.process([op, prod], 6, 0.0, ctx)
    assert det.process([op, prod], 12, 1.2, ctx) == []


def test_stepping_is_silent_without_a_ground_plane_fit():
    op = build_track([(600, 300)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 300.0), fps=FPS, track_id=3)
    prod = build_track([(600, 500)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    det = SteppingDetector()
    assert det.process([op, prod], 12, 1.2, base_context(ground_plane=None)) == []


# --------------------------------------------------------------- stacking
def test_stacking_fires_on_persistent_overhang():
    top = build_track([(600, 300, 260.0, 100.0)] * 6, fps=FPS, track_id=1)
    bot = build_track([(600, 400, 140.0, 100.0)] * 6, fps=FPS, track_id=2)
    # Top rests on bottom: top y2 == bottom y1.
    top.box = [470.0, 300.0, 730.0, 350.0]
    bot.box = [530.0, 350.0, 670.0, 450.0]
    top.width, bot.width = 260.0, 140.0
    ctx = base_context()

    det = StackingDetector()
    det.process([top, bot], 1, 0.0, ctx)          # start the stability timer
    events = det.process([top, bot], 30, 3.0, ctx)
    assert len(events) == 1
    assert events[0].metadata["width_ratio"] > 1.25


def test_stacking_ignores_a_transient_overlap():
    top = build_track([(600, 300, 260.0, 100.0)] * 6, fps=FPS, track_id=1)
    bot = build_track([(600, 400, 140.0, 100.0)] * 6, fps=FPS, track_id=2)
    top.box = [470.0, 300.0, 730.0, 350.0]
    bot.box = [530.0, 350.0, 670.0, 450.0]
    ctx = base_context()
    det = StackingDetector()
    det.process([top, bot], 1, 0.0, ctx)
    assert det.process([top, bot], 5, 0.5, ctx) == []  # under the 1.5 s hold


# ------------------------------------------------------------- orientation
def test_orientation_requires_an_observed_upright_to_flat_transition():
    path = [(600, 400, 90.0, 260.0)] * 5 + [(600, 400, 260.0, 90.0)] * 5
    trk = build_track(path, fps=FPS)
    ctx = base_context()
    det = OrientationDetector()
    det.process([trk], 10, 1.0, ctx)
    events = det.process([trk], 40, 4.0, ctx)
    assert len(events) == 1
    assert events[0].metadata["observed_transition"] is True


def test_orientation_ignores_an_item_that_was_always_flat():
    """Most packages are wider than tall from a ceiling camera; that is normal."""
    trk = build_track([(600, 400, 260.0, 90.0)] * 12, fps=FPS)
    det = OrientationDetector()
    det.process([trk], 10, 1.0, base_context())
    assert det.process([trk], 40, 4.0, base_context()) == []


# ------------------------------------------------------------ scene-gated
def test_wet_floor_is_silent_when_the_condition_is_not_declared():
    trk = build_track([(200 + i * 26, 660) for i in range(12)], fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 12)
    assert WetFloorDetector().process([trk], 12, 1.2, base_context(wet_floor_active=False)) == []


def test_wet_floor_fires_when_declared_and_goods_move_through_it():
    trk = build_track([(200 + i * 26, 660) for i in range(12)], fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 12)
    ctx = base_context(wet_floor_active=True)
    det = WetFloorDetector()
    det.process([trk], 12, 0.0, ctx)
    events = det.process([trk], 30, 2.0, ctx)
    assert len(events) == 1


def test_dock_hazard_is_silent_when_camera_is_not_a_dock():
    trk = build_track([(200 + i * 30, 640, 300.0, 260.0) for i in range(12)], fps=FPS)
    _set_states(trk, [MotionState.SLIDING] * 12)
    assert DockDetector().process([trk], 12, 1.2, base_context(dock_transfer_active=False)) == []


def test_designated_area_is_silent_without_a_configured_zone():
    trk = build_track([(600, 400)] * 12, fps=FPS)
    _set_states(trk, [MotionState.SETTLED] * 12, contact=False)
    assert DesignatedAreaDetector().process([trk], 12, 5.0, base_context(staging_zone=None)) == []


def test_designated_area_fires_outside_a_configured_zone():
    trk = build_track([(1200, 690)] * 12, fps=FPS)  # far right, outside the polygon
    _set_states(trk, [MotionState.SETTLED] * 12, contact=False)
    zone = [[0.0, 0.0], [0.4, 0.0], [0.4, 1.0], [0.0, 1.0]]
    ctx = base_context(staging_zone=zone)
    det = DesignatedAreaDetector()
    det.process([trk], 1, 0.0, ctx)
    events = det.process([trk], 60, 6.0, ctx)
    assert len(events) == 1


# ------------------- detection dropout must not reset contact timers --------
def test_stepping_survives_intermittent_product_detection():
    """
    Regression: the contact timer was deleted whenever a pair was missing from
    a single frame. With ~59% product detection on the pilot footage the 0.6 s
    dwell could never accumulate, so a clip that clearly shows an operator
    standing on a carton produced no event at all.
    """
    op = build_track([(600, 720 * 0.70 - 162)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 324.0), fps=FPS, track_id=3)
    prod = build_track([(600, 720 * 0.76)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())

    det = SteppingDetector()
    det.process([op, prod], 6, 0.0, ctx)        # contact starts
    det.process([op, prod], 9, 0.2, ctx)
    det.process([op], 12, 0.4, ctx)             # product detection drops out
    events = det.process([op, prod], 15, 0.7, ctx)  # reappears, still one contact
    assert len(events) == 1, "a one-frame dropout must not reset the dwell timer"


def test_stepping_contact_expires_after_a_real_absence():
    """The grace window must not let an operator who genuinely stepped off
    accumulate dwell across separate visits."""
    op = build_track([(600, 720 * 0.70 - 162)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 324.0), fps=FPS, track_id=3)
    prod = build_track([(600, 720 * 0.76)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())

    det = SteppingDetector()
    det.process([op, prod], 6, 0.0, ctx)
    det.process([op], 30, 3.0, ctx)             # absent far longer than the grace
    # Contact restarts here, so 0.2 s later the dwell is not yet satisfied.
    det.process([op, prod], 36, 3.2, ctx)
    assert det.process([op, prod], 39, 3.4, ctx) == []


def test_stacking_survives_intermittent_detection():
    """Same dropout flaw affected the 1.5 s stacking stability requirement."""
    top = build_track([(600, 400)] * 6, size=(260.0, 90.0), fps=FPS, track_id=1)
    bot = build_track([(600, 490)] * 6, size=(180.0, 90.0), fps=FPS, track_id=2)
    ctx = base_context()

    det = StackingDetector()
    det.process([top, bot], 3, 0.0, ctx)
    det.process([top], 6, 0.3, ctx)             # bottom package drops out
    events = det.process([top, bot], 12, 1.8, ctx)
    assert len(events) == 1, "a dropout must not reset the stacking stability timer"


def test_stepping_requires_more_than_a_single_frame_coincidence():
    """The observation count is what rejects a one-frame overlap."""
    op = build_track([(600, 720 * 0.70 - 162)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 324.0), fps=FPS, track_id=3)
    prod = build_track([(600, 720 * 0.76)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())

    det = SteppingDetector()
    assert det.process([op, prod], 6, 0.0, ctx) == []
    assert det.process([op, prod], 9, 0.3, ctx) == []   # still below MIN_OBSERVATIONS


def test_stepping_is_silent_when_the_ground_plane_fit_is_too_noisy():
    """
    A very noisy fit makes the required elevation exceed what stepping on a
    package can physically produce. The detector must say it cannot judge
    rather than appearing active while being unable to fire.
    """
    op = build_track([(600, 720 * 0.70 - 162)] * 6, entity=WarehouseEntity.OPERATOR,
                     size=(90.0, 324.0), fps=FPS, track_id=3)
    prod = build_track([(600, 720 * 0.76)] * 6, size=(200.0, 90.0), fps=FPS, track_id=7)
    ctx = base_context(ground_plane=_ground_plane())
    ctx["ground_plane_residual"] = 0.09   # 2.5 * 0.09 = 0.225 > MAX_ELEVATION

    det = SteppingDetector()
    assert det.process([op, prod], 6, 0.0, ctx) == []
    assert det.unable_to_judge is not None
    assert "too noisy" in det.unable_to_judge
