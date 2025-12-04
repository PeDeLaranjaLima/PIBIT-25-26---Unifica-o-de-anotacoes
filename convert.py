from __future__ import annotations

from voc_to_yolo import VOCtoYOLO
from coco_to_yolo import COCOtoYOLO
from yolo_to_coco import YOLOtoCOCO

class convert:
    def __init__(self) -> None:
        self.VOCtoYOLO = VOCtoYOLO
        self.COCOtoYOLO = COCOtoYOLO
        self.YOLOtoCOCO = YOLOtoCOCO

"""

EXEPLO DE USO 

    # Caminho base pode ser relativo ou absoluto
    base_dir = Path("CPLID_val") / "val"

    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    output_file = Path("saida.json")

    conv = convert()

    coco = conv.YOLOtoCOCOConverter(
        images_dir,
        labels_dir,
        ["013"],
        output_file,
    )

    coco.convert()

"""