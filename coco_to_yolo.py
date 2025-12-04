from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

class COCOtoYOLO:
    def __init__(self, json_path: str | Path, output_dir: str | Path) -> None:
        self.json_path = Path(json_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.coco_data: dict = json.load(f)

        self.images: Dict[int, dict] = {
            img["id"]: img for img in self.coco_data["images"]
        }

        self.categories: Dict[int, str] = {
            cat["id"]: cat["name"] for cat in self.coco_data["categories"]
        }

        self.class_map: Dict[str, int] = {
            name: i for i, name in enumerate(sorted(self.categories.values()))
        }

    def convert_bbox(
        self, bbox: List[float], img_width: int, img_height: int
    ) -> Tuple[float, float, float, float]:

        x, y, w, h = bbox

        return (
            (x + w / 2) / img_width,
            (y + h / 2) / img_height,
            w / img_width,
            h / img_height,
        )

    def convert(self) -> None:
        annotations_by_image: Dict[int, list] = {}

        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            annotations_by_image.setdefault(img_id, []).append(ann)

        for img_id, anns in annotations_by_image.items():
            image_info = self.images[img_id]
            img_w = image_info["width"]
            img_h = image_info["height"]
            img_name = Path(image_info["file_name"]).stem

            yolo_lines: List[str] = []

            for ann in anns:
                cat_name = self.categories[ann["category_id"]]
                cls_id = self.class_map[cat_name]

                bb = self.convert_bbox(ann["bbox"], img_w, img_h)
                yolo_lines.append(f"{cls_id} {' '.join(map(str, bb))}")

            txt_path = self.output_dir / f"{img_name}.txt"
            txt_path.write_text("\n".join(yolo_lines))

        print(f"Conversão concluída! Arquivos YOLO salvos em: {self.output_dir}")
