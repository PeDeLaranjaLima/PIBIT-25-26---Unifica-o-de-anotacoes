from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


class YOLOtoCOCO:
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: list[str],
        output_json: str | Path,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.output_json = Path(output_json)

    def convert_bbox(
        self,
        bbox: list[float],
        img_width: int,
        img_height: int,
    ) -> list[float]:
        x_center, y_center, w, h = bbox

        x_min = (x_center - w / 2) * img_width
        y_min = (y_center - h / 2) * img_height

        return [x_min, y_min, w * img_width, h * img_height]

    def convert(self) -> None:
        images: list[dict] = []
        annotations: list[dict] = []
        categories: list[dict] = []

        ann_id = 1
        img_id = 1

        for i, name in enumerate(self.class_names):
            categories.append(
                {
                    "id": i + 1,
                    "name": name,
                    "supercategory": "object",
                }
            )

        for label_file in self.labels_dir.glob("*.txt"):
            image_name = label_file.stem

            img_path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = self.images_dir / f"{image_name}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

            if img_path is None:
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            images.append(
                {
                    "id": img_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                }
            )

            for line in label_file.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id, x, y, w, h = map(float, parts)
                coco_box = self.convert_bbox([x, y, w, h], width, height)

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(class_id) + 1,
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0,
                    }
                )

                ann_id += 1

            img_id += 1

        coco_output = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        self.output_json.write_text(json.dumps(coco_output, indent=4))

        print(
            f"Conversão concluída! Arquivo COCO salvo em: {self.output_json}"
        )
