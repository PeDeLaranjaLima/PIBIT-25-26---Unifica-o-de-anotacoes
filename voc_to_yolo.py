from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

class VOCtoYOLO:
    def __init__(self, class_map: Dict[str, int]) -> None:
        self.class_map = class_map

    @staticmethod
    def convert_box(
        size: Tuple[int, int],
        box: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:

        dw = 1.0 / size[0]
        dh = 1.0 / size[1]

        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]

        return x * dw, y * dh, w * dw, h * dh

    def convert_xml(self, xml_path: str | Path) -> List[str]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        yolo_lines: List[str] = []

        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in self.class_map:
                continue

            cls_id = self.class_map[cls]
            xmlbox = obj.find("bndbox")

            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )

            bb = self.convert_box((w, h), b)
            yolo_lines.append(f"{cls_id} {' '.join(map(str, bb))}")

        return yolo_lines
