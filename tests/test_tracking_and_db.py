"""Tracker kinematics / state machine, and direct database-layer behaviour."""

import pytest

from backend.database.db import DatabaseManager, get_connection
from detection.object_classes import WarehouseEntity
from detection.tracker import MotionState, PersistentTracker, compute_iou
from helpers import FRAME_H, FRAME_W, make_detection, seed_incident, seed_video


# ---------------------------------------------------------------- geometry
def test_iou_of_identical_and_disjoint_boxes():
    assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert compute_iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0
    assert 0 < compute_iou([0, 0, 10, 10], [5, 5, 15, 15]) < 1


# ----------------------------------------------------------------- tracking
def _tracker():
    return PersistentTracker(frame_height=FRAME_H, frame_width=FRAME_W)


def test_track_id_is_stable_across_frames():
    tr = _tracker()
    ids = []
    for i in range(12):
        det = make_detection(400 + i * 6, 400, 120, 100, WarehouseEntity.CARTON)
        tracks = tr.update([det], i, i / 10.0, 10.0)
        ids.append(tracks[0].track_id)
    assert len(set(ids)) == 1, "a smoothly moving object must keep one track id"
    assert tr.tracks[ids[0]].hits == 12


def test_separate_objects_get_separate_ids():
    tr = _tracker()
    for i in range(6):
        tr.update(
            [
                make_detection(200, 300, 100, 100, WarehouseEntity.CARTON),
                make_detection(900, 600, 100, 100, WarehouseEntity.CARTON),
            ],
            i, i / 10.0, 10.0,
        )
    assert len(tr.tracks) == 2


def test_operators_and_products_never_share_a_track():
    tr = _tracker()
    for i in range(6):
        tr.update([make_detection(400, 400, 100, 260, WarehouseEntity.OPERATOR)], i, i / 10.0, 10.0)
    before = set(tr.tracks)
    # A product appearing at the same place must not inherit the operator track.
    tr.update([make_detection(400, 400, 100, 260, WarehouseEntity.CARTON)], 6, 0.6, 10.0)
    products = [t for t in tr.tracks.values() if t.entity_type is WarehouseEntity.CARTON]
    assert len(products) == 1
    assert products[0].track_id not in before


def test_lost_track_is_dropped_after_the_grace_period():
    tr = PersistentTracker(max_lost_frames=3, frame_height=FRAME_H, frame_width=FRAME_W)
    for i in range(5):
        tr.update([make_detection(400, 400, 100, 100, WarehouseEntity.CARTON)], i, i / 10.0, 10.0)
    assert len(tr.tracks) == 1
    for i in range(5, 12):
        tr.update([], i, i / 10.0, 10.0)
    assert len(tr.tracks) == 0


def test_velocity_is_resolution_independent():
    """The same physical motion must yield the same normalised velocity."""
    speeds = []
    for h, w in [(720.0, 1280.0), (1440.0, 2560.0)]:
        tr = PersistentTracker(frame_height=h, frame_width=w)
        for i in range(8):
            # Move a constant fraction (2%) of frame height per step.
            det = make_detection(0.3 * w, (0.2 + i * 0.02) * h, 0.1 * w, 0.1 * h,
                                 WarehouseEntity.CARTON)
            tracks = tr.update([det], i, i / 10.0, 10.0)
        speeds.append(tracks[0].vy)
    assert speeds[0] == pytest.approx(speeds[1], rel=0.02)


def test_falling_state_is_set_on_sustained_descent():
    tr = _tracker()
    for i in range(8):
        det = make_detection(600, 150 + i * 70, 120, 100, WarehouseEntity.CARTON)
        tracks = tr.update([det], i, i / 10.0, 10.0)
    assert tracks[0].state is MotionState.FALLING


def test_settled_state_follows_motion_then_rest():
    tr = _tracker()
    for i in range(6):
        tr.update([make_detection(300 + i * 60, 600, 120, 100, WarehouseEntity.CARTON)],
                  i, i / 10.0, 10.0)
    for i in range(6, 16):
        tracks = tr.update([make_detection(600, 600, 120, 100, WarehouseEntity.CARTON)],
                           i, i / 10.0, 10.0)
    assert tracks[0].state is MotionState.SETTLED


def test_operator_contact_is_recorded_for_a_nearby_product():
    tr = _tracker()
    for i in range(6):
        tracks = tr.update(
            [
                make_detection(600, 500, 90, 300, WarehouseEntity.OPERATOR),
                make_detection(640, 560, 110, 100, WarehouseEntity.CARTON),
            ],
            i, i / 10.0, 10.0,
        )
    carton = next(t for t in tracks if t.entity_type is WarehouseEntity.CARTON)
    assert carton.operator_contact_id is not None


def test_ground_plane_is_none_until_depth_varies():
    tr = _tracker()
    for i in range(40):
        tr.update([make_detection(600, 600, 90, 300, WarehouseEntity.OPERATOR)], i, i / 10.0, 10.0)
    # Every observation at the same depth: no usable gradient, so no fit.
    assert tr.expected_floor_y(0.4) is None


def test_ground_plane_fits_when_operators_appear_at_different_depths():
    tr = _tracker()
    for i in range(40):
        # Alternate a near (tall) and a far (short) operator.
        near = make_detection(400, 600, 90, 300, WarehouseEntity.OPERATOR)
        far = make_detection(900, 380, 40, 120, WarehouseEntity.OPERATOR)
        tr.update([near, far], i, i / 10.0, 10.0)
    tall = tr.expected_floor_y(300 / FRAME_H)
    short = tr.expected_floor_y(120 / FRAME_H)
    assert tall is not None and short is not None
    assert tall > short, "nearer (taller) operators stand lower in the frame"


# ---------------------------------------------------------------- database
def test_incident_round_trips_with_json_fields_intact():
    seed_video()
    seed_incident()
    got = DatabaseManager.get_incident_by_id("inc_test01")
    assert got["bounding_box"] == [10.0, 20.0, 110.0, 140.0]
    assert got["risk_factors"][0]["points"] == 18.0
    assert got["evidence_stages"][1]["stage"] == "falling"


def test_saving_the_same_incident_twice_does_not_duplicate_it():
    seed_video()
    seed_incident()
    seed_incident(risk_level="CRITICAL")
    assert DatabaseManager.count_incidents() == 1
    assert DatabaseManager.get_incident_by_id("inc_test01")["risk_level"] == "CRITICAL"


def test_filters_compose():
    seed_video()
    seed_incident(id="a", behaviour_type="product_drop", risk_level="HIGH", bay="Dock 01")
    seed_incident(id="b", behaviour_type="product_drop", risk_level="LOW", bay="Dock 02")
    seed_incident(id="c", behaviour_type="product_drag", risk_level="HIGH", bay="Dock 01")
    assert len(DatabaseManager.get_incidents(behaviour_type="product_drop")) == 2
    assert len(DatabaseManager.get_incidents(risk_level="HIGH", bay="Dock 01")) == 2
    assert len(DatabaseManager.get_incidents(behaviour_type="product_drop", risk_level="HIGH")) == 1


def test_pagination_returns_distinct_pages():
    seed_video()
    for i in range(10):
        seed_incident(id=f"inc_{i}", timestamp_sec=float(i))
    page1 = DatabaseManager.get_incidents(limit=4, offset=0)
    page2 = DatabaseManager.get_incidents(limit=4, offset=4)
    assert len(page1) == 4 and len(page2) == 4
    assert not ({i["id"] for i in page1} & {i["id"] for i in page2})


def test_recurrence_history_is_scoped_by_bay():
    seed_video()
    seed_incident(id="a", behaviour_type="product_drag", bay="Dock 01")
    seed_incident(id="b", behaviour_type="product_drag", bay="Dock 02")
    assert DatabaseManager.get_behaviour_history("product_drag") == 2
    assert DatabaseManager.get_behaviour_history("product_drag", bay="Dock 01") == 1


def test_deleting_a_video_cascades_to_its_incidents():
    seed_video()
    seed_incident()
    assert DatabaseManager.delete_video("vid_test01") == 1
    assert DatabaseManager.count_incidents() == 0


def test_migrations_are_idempotent():
    from backend.database.db import init_db

    init_db()
    init_db()
    with get_connection() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(incidents)")}
    assert {"camera_id", "bay", "shift", "risk_factors", "evidence_stages",
            "evidence_clip_path", "review_status"} <= cols
