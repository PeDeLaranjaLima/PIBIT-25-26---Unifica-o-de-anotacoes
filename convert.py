from __future__ import annotations

from voc_to_yolo import VOCtoYOLO
from coco_to_yolo import COCOtoYOLO
from yolo_to_coco import YOLOtoCOCO

class convert:
    def __init__(self) -> None:
        self.VOCtoYOLO = VOCtoYOLO
        self.COCOtoYOLO = COCOtoYOLO
        self.YOLOtoCOCO = YOLOtoCOCO
