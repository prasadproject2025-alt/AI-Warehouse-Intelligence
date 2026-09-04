"""
Warehouse Object Taxonomy & Mapping.

Two detector backends feed the same taxonomy:

* ``open_vocab`` - YOLO-World prompted with warehouse nouns. COCO has no
  "carton" or "pallet" class, so this is the only backend that can actually
  see the products being handled.
* ``coco`` - plain YOLOv8 fallback. Only classes with a defensible warehouse
  meaning are accepted; there is deliberately NO catch-all mapping, because
  mapping every unknown COCO class (kite, umbrella, skis...) onto "carton" was
  the single largest source of false incidents in the first implementation.
"""

from enum import Enum
from typing import Dict, List, Optional


class WarehouseEntity(str, Enum):
    OPERATOR = "operator"
    CARTON = "carton"
    CUPBOARD = "cupboard"
    MATTRESS = "mattress"
    PALLET = "pallet"
    TROLLEY = "trolley"
    EQUIPMENT = "equipment"
    VEHICLE = "vehicle"


#: Entities that represent goods being handled (behaviour subjects).
PRODUCT_ENTITIES = {
    WarehouseEntity.CARTON,
    WarehouseEntity.CUPBOARD,
    WarehouseEntity.MATTRESS,
}

#: Entities that count as approved material-handling equipment.
HANDLING_EQUIPMENT_ENTITIES = {
    WarehouseEntity.TROLLEY,
    WarehouseEntity.PALLET,
    WarehouseEntity.EQUIPMENT,
}

#: Text prompts given to the open-vocabulary detector, in prompt order.
#: Index order matters: the model returns class indices into this list.
OPEN_VOCAB_PROMPTS: List[str] = [
    "person",
    "box",
    "carton",
    "package",
    "mattress",
    "pallet",
    "trolley",
    "forklift",
    "truck",
]

OPEN_VOCAB_TO_WAREHOUSE: Dict[str, WarehouseEntity] = {
    "person": WarehouseEntity.OPERATOR,
    "box": WarehouseEntity.CARTON,
    "carton": WarehouseEntity.CARTON,
    "package": WarehouseEntity.CARTON,
    "mattress": WarehouseEntity.MATTRESS,
    "pallet": WarehouseEntity.PALLET,
    "trolley": WarehouseEntity.TROLLEY,
    "forklift": WarehouseEntity.EQUIPMENT,
    "truck": WarehouseEntity.VEHICLE,
}

#: Strict COCO whitelist for the fallback backend. Anything not listed is
#: discarded rather than guessed at.
COCO_TO_WAREHOUSE: Dict[str, WarehouseEntity] = {
    "person": WarehouseEntity.OPERATOR,
    "suitcase": WarehouseEntity.CARTON,
    "backpack": WarehouseEntity.CARTON,
    "handbag": WarehouseEntity.CARTON,
    "refrigerator": WarehouseEntity.CUPBOARD,
    "couch": WarehouseEntity.MATTRESS,
    "bed": WarehouseEntity.MATTRESS,
    "truck": WarehouseEntity.VEHICLE,
    "bus": WarehouseEntity.VEHICLE,
    "car": WarehouseEntity.VEHICLE,
}

# Colour palette (BGR for OpenCV) used for overlays.
ENTITY_COLORS = {
    WarehouseEntity.OPERATOR: (255, 178, 50),
    WarehouseEntity.CARTON: (0, 215, 255),
    WarehouseEntity.CUPBOARD: (200, 100, 255),
    WarehouseEntity.MATTRESS: (180, 220, 0),
    WarehouseEntity.PALLET: (128, 128, 128),
    WarehouseEntity.TROLLEY: (255, 140, 0),
    WarehouseEntity.EQUIPMENT: (0, 165, 255),
    WarehouseEntity.VEHICLE: (100, 100, 200),
}


def map_label(label: str, backend: str = "open_vocab") -> Optional[WarehouseEntity]:
    """
    Map a detector label onto the warehouse taxonomy.

    Returns ``None`` for labels with no defensible warehouse meaning so the
    caller can drop the detection instead of inventing an entity for it.
    """
    cleaned = label.lower().strip()
    table = OPEN_VOCAB_TO_WAREHOUSE if backend == "open_vocab" else COCO_TO_WAREHOUSE
    return table.get(cleaned)


def refine_product_entity(
    entity: WarehouseEntity, box: List[float]
) -> WarehouseEntity:
    """
    Geometric refinement for products: a tall, narrow, large carton in this
    domain is a knock-down furniture / cupboard package rather than a small box.
    """
    if entity is not WarehouseEntity.CARTON:
        return entity
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    if h / w > 1.6 and h > 180:
        return WarehouseEntity.CUPBOARD
    return entity
