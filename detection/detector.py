"""
Warehouse Object Detector
Wraps Ultralytics YOLO with warehouse entity mapping and geometric package analysis.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from ultralytics import YOLO
from detection.object_classes import WarehouseEntity, map_coco_class

class Detection:
    def __init__(
        self,
        box: List[float],
        confidence: float,
        raw_class: str,
        entity_type: WarehouseEntity,
        track_id: Optional[int] = None
    ):
        self.box = [float(b) for b in box] # [x1, y1, x2, y2]
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
            "track_id": self.track_id
        }

class WarehouseDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(model_path)
        
        # Relevant COCO class IDs: person (0), backpack (24), suitcase (28), handbag (26), 
        # bottle (39), chair (56), couch (57), bed (59), refrigerator (72), tv (62), truck (7)
        self.relevant_coco_classes = {0, 7, 24, 26, 28, 56, 57, 59, 62, 72}

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on frame and return list of Detection objects.
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]

        detections: List[Detection] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            # Accept if relevant class or if confidence is reasonable
            raw_class = self.model.names[cls_id]
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()

            entity = map_coco_class(raw_class, xyxy)
            
            # Filter out tiny spurious bounding boxes (e.g. noise < 20x20 pixels)
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            if w < 25 or h < 25:
                continue

            # In warehouse footage, TV / large screens detected in loading dock are usually cartons/pallets
            if raw_class == "tv" and (w > 100 or h > 100):
                entity = WarehouseEntity.CARTON

            detections.append(Detection(
                box=xyxy,
                confidence=conf,
                raw_class=raw_class,
                entity_type=entity
            ))

        return detections
