"""
Warehouse Object Detector.

Wraps Ultralytics with two selectable backends and a strict warehouse taxonomy:

* ``open_vocab`` (default) - YOLO-World prompted with warehouse nouns so that
  cartons, mattresses, pallets and trolleys are actually detectable. COCO
  contains none of those classes.
* ``coco`` - YOLOv8 fallback used when the open-vocabulary weights or the CLIP
  text encoder are unavailable. Only whitelisted COCO classes are kept.

Per-class confidence thresholds are applied because products score noticeably
lower than people under an open-vocabulary head.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

import config
from detection.object_classes import (
    OPEN_VOCAB_PROMPTS,
    PRODUCT_ENTITIES,
    WarehouseEntity,
    map_label,
    refine_product_entity,
)

logger = logging.getLogger(__name__)


class Detection:
    __slots__ = ("box", "confidence", "raw_class", "entity_type", "track_id")

    def __init__(
        self,
        box: List[float],
        confidence: float,
        raw_class: str,
        entity_type: WarehouseEntity,
        track_id: Optional[int] = None,
    ) -> None:
        self.box = [float(b) for b in box]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.raw_class = raw_class
        self.entity_type = entity_type
        self.track_id = track_id

    @property
    def center(self) -> List[float]:
        return [(self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0]

    @property
    def width(self) -> float:
        return max(1.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(b, 1) for b in self.box],
            "center": [round(c, 1) for c in self.center],
            "confidence": round(self.confidence, 3),
            "raw_class": self.raw_class,
            "entity_type": self.entity_type.value,
            "track_id": self.track_id,
        }


def _iou(a: List[float], b: List[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class WarehouseDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        prefer_open_vocab: Optional[bool] = None,
    ) -> None:
        self.iou_threshold = (
            config.IOU_THRESHOLD if iou_threshold is None else float(iou_threshold)
        )
        # Lowest of the per-class thresholds: the model-level filter must not
        # discard product boxes before the per-class gate can see them.
        self.person_conf = config.PERSON_CONF
        self.product_conf = config.PRODUCT_CONF if conf_threshold is None else float(conf_threshold)
        self.equipment_conf = config.EQUIPMENT_CONF
        self._base_conf = min(self.person_conf, self.product_conf, self.equipment_conf)

        use_ov = config.USE_OPEN_VOCAB if prefer_open_vocab is None else prefer_open_vocab
        self.backend = "coco"
        self.model = None

        if use_ov and model_path is None:
            self.model = self._try_open_vocab()

        if self.model is None:
            from ultralytics import YOLO

            path = model_path or config.FALLBACK_MODEL_PATH
            self.model = YOLO(path)
            self.backend = "coco"
            logger.info("Detector backend: COCO YOLO (%s)", path)

    def _try_open_vocab(self):
        """Load YOLO-World with the warehouse prompt set, or return None."""
        try:
            from ultralytics import YOLOWorld

            model = YOLOWorld(config.OPEN_VOCAB_MODEL_PATH)
            model.set_classes(OPEN_VOCAB_PROMPTS)
            self.backend = "open_vocab"
            logger.info(
                "Detector backend: open-vocabulary YOLO-World (%s) with prompts %s",
                config.OPEN_VOCAB_MODEL_PATH,
                OPEN_VOCAB_PROMPTS,
            )
            return model
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            logger.warning(
                "Open-vocabulary detector unavailable (%s); falling back to COCO YOLO. "
                "Cartons and pallets will NOT be detectable in this mode.",
                exc,
            )
            return None

    # ------------------------------------------------------------------ infer
    def _class_threshold(self, entity: WarehouseEntity) -> float:
        if entity is WarehouseEntity.OPERATOR:
            return self.person_conf
        if entity in PRODUCT_ENTITIES:
            return self.product_conf
        return self.equipment_conf

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one frame and return taxonomy-mapped detections."""
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        min_side = max(12.0, 0.012 * min(h, w))  # resolution-relative noise floor

        results = self.model.predict(
            frame,
            imgsz=config.INFERENCE_IMGSZ,
            conf=self._base_conf,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        names = results.names if getattr(results, "names", None) else self.model.names
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            raw_class = names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]

            entity = map_label(raw_class, self.backend)
            if entity is None:
                continue  # no defensible warehouse meaning - drop it
            if conf < self._class_threshold(entity):
                continue

            bw = xyxy[2] - xyxy[0]
            bh = xyxy[3] - xyxy[1]
            if bw < min_side or bh < min_side:
                continue
            # A single box covering nearly the whole frame is a scene-level
            # false positive, not a handled product.
            if entity is not WarehouseEntity.VEHICLE and (bw * bh) > 0.72 * (w * h):
                continue

            if entity in PRODUCT_ENTITIES:
                entity = refine_product_entity(entity, xyxy)

            detections.append(
                Detection(box=xyxy, confidence=conf, raw_class=raw_class, entity_type=entity)
            )

        return self._suppress_cross_class_duplicates(detections)

    @staticmethod
    def _suppress_cross_class_duplicates(dets: List[Detection]) -> List[Detection]:
        """
        Open-vocabulary prompts overlap ("box"/"carton"/"package" all fire on the
        same object). Keep the highest-confidence detection per physical object.
        """
        ordered = sorted(dets, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []
        for det in ordered:
            duplicate = False
            for k in kept:
                if _iou(det.box, k.box) > 0.62:
                    # Operators and products never merge into one another.
                    same_group = (det.entity_type is WarehouseEntity.OPERATOR) == (
                        k.entity_type is WarehouseEntity.OPERATOR
                    )
                    if same_group:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(det)
        return kept
