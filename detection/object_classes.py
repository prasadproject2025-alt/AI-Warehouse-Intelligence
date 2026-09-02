"""
Warehouse Object Taxonomy & Mapping
Defines standardized entities for warehouse video intelligence.
"""

from enum import Enum
from typing import Dict, List, Set

class WarehouseEntity(str, Enum):
    OPERATOR = "operator"
    CARTON = "carton"
    CUPBOARD = "cupboard"
    MATTRESS = "mattress"
    PALLET = "pallet"
    TROLLEY = "trolley"
    EQUIPMENT = "equipment"
    VEHICLE = "vehicle"
    UNKNOWN = "product"

# Mapping from COCO pretrained classes to warehouse entities
COCO_TO_WAREHOUSE: Dict[str, WarehouseEntity] = {
    "person": WarehouseEntity.OPERATOR,
    "suitcase": WarehouseEntity.CARTON,
    "backpack": WarehouseEntity.CARTON,
    "handbag": WarehouseEntity.CARTON,
    "box": WarehouseEntity.CARTON,
    "refrigerator": WarehouseEntity.CUPBOARD,
    "couch": WarehouseEntity.MATTRESS,
    "bed": WarehouseEntity.MATTRESS,
    "truck": WarehouseEntity.VEHICLE,
    "car": WarehouseEntity.VEHICLE,
}

# Color palette (BGR for OpenCV) for visual overlays
ENTITY_COLORS = {
    WarehouseEntity.OPERATOR: (255, 178, 50),     # Bright Blue/Amber
    WarehouseEntity.CARTON: (0, 215, 255),       # Gold/Yellow
    WarehouseEntity.CUPBOARD: (200, 100, 255),   # Magenta
    WarehouseEntity.MATTRESS: (180, 220, 0),     # Cyan/Lime
    WarehouseEntity.PALLET: (128, 128, 128),     # Gray
    WarehouseEntity.TROLLEY: (255, 140, 0),      # Deep Orange
    WarehouseEntity.EQUIPMENT: (0, 165, 255),    # Orange
    WarehouseEntity.VEHICLE: (100, 100, 200),    # Slate
    WarehouseEntity.UNKNOWN: (200, 200, 200)
}

def map_coco_class(coco_label: str, bbox: List[float] = None) -> WarehouseEntity:
    """
    Map COCO detection label to warehouse entity taxonomy.
    Can also use bounding box geometric heuristics (aspect ratio, height)
    to refine classification.
    """
    cleaned = coco_label.lower().strip()
    if cleaned in COCO_TO_WAREHOUSE:
        entity = COCO_TO_WAREHOUSE[cleaned]
        # Geometric refinement: if carton is very tall (aspect ratio h/w > 1.8), it may be a vertical cabinet/cupboard
        if entity == WarehouseEntity.CARTON and bbox is not None:
            x1, y1, x2, y2 = bbox
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            if h / w > 1.8 and h > 200:
                return WarehouseEntity.CUPBOARD
        return entity
    return WarehouseEntity.CARTON
