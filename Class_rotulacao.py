import xml.etree.ElementTree as ET
import json
import os
from PIL import Image

class convert:

    def __init__(self): # Permitir instância que expõe as classes internas
        self.VOCtoYOLOConverter = self.VOCtoYOLOConverter
        self.COCOtoYOLOConverter = self.COCOtoYOLOConverter
        self.YOLOtoCOCOConverter = self.YOLOtoCOCOConverter

    class VOCtoYOLOConverter:
        """
        Classe responsável por converter anotações no formato PASCAL VOC (.xml)
        para o formato YOLO (.txt).
        """

        def __init__(self, class_map):
            """
            Inicializa o conversor com o mapeamento de classes.

            Args:
                class_map (dict): Dicionário que mapeia nomes de classes para IDs (ex: {'insulator': 0}).
            """
            self.class_map = class_map

        @staticmethod
        def convert_box(size, box):
            """
            Converte uma bounding box PASCAL VOC para o formato YOLO.

            Args:
                size (tuple): (largura, altura) da imagem.
                box (tuple): (xmin, xmax, ymin, ymax)

            Returns:
                tuple: (x_centro, y_centro, largura, altura) em formato YOLO.
            """
            dw = 1.0 / size[0]
            dh = 1.0 / size[1]
            x = (box[0] + box[1]) / 2.0
            y = (box[2] + box[3]) / 2.0
            w = box[1] - box[0]
            h = box[3] - box[2]
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            return (x, y, w, h)

        def convert_xml(self, xml_path):
            """
            Lê um arquivo .xml do PASCAL VOC e retorna o conteúdo convertido em formato YOLO.

            Args:
                xml_path (str): Caminho para o arquivo XML.

            Returns:
                list[str]: Linhas em formato YOLO (ex: ["0 0.5 0.5 0.3 0.4"])
            """
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            yolo_lines = []
            for obj in root.iter('object'):
                cls = obj.find('name').text
                if cls not in self.class_map:
                    continue
                cls_id = self.class_map[cls]
                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text)
                )
                bb = self.convert_box((w, h), b)
                line = f"{cls_id} {' '.join(map(str, bb))}"
                yolo_lines.append(line)
            return yolo_lines


    """
    # Uso 

    yolo_labels = converter.convert_xml("/arquivo.xml") // caminho

    # Salvar em arquivo .txt, talvez seja melhor
    with open("saida.txt", "w") as f:
        f.write("\n".join(yolo_labels))


    """
    class COCOtoYOLOConverter:
        """
        Classe responsável por converter anotações no formato COCO (JSON)
        para o formato YOLO (.txt).
        """

        def __init__(self, json_path, output_dir):
            """
            Inicializa o conversor.

            Args:
                json_path (str): Caminho para o arquivo COCO .json (ex: instances_train.json).
                output_dir (str): Pasta onde os arquivos YOLO serão salvos.
            """
            self.json_path = json_path
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

            # Carregar JSON
            with open(self.json_path, 'r') as f:
                self.coco_data = json.load(f)

            # Criar dicionários auxiliares
            self.images = {img['id']: img for img in self.coco_data['images']}
            self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

            # Mapeia classes para IDs sequenciais (0, 1, 2, ...)
            self.class_map = {name: i for i, name in enumerate(sorted(self.categories.values()))}

        def convert_bbox(self, bbox, img_width, img_height):
            """
            Converte bounding box do formato COCO para YOLO.

            Args:
                bbox (list): [x_min, y_min, width, height] (em pixels absolutos)
                img_width (int): Largura da imagem
                img_height (int): Altura da imagem

            Returns:
                tuple: (x_center, y_center, width, height) normalizados
            """
            x, y, w, h = bbox
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            return x_center, y_center, w_norm, h_norm

        def convert(self):
            """
            Converte todas as anotações COCO para YOLO e salva um .txt por imagem.
            """
            # Agrupar anotações por imagem
            annotations_by_image = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                annotations_by_image.setdefault(img_id, []).append(ann)

            # Converter imagem por imagem
            for img_id, anns in annotations_by_image.items():
                image_info = self.images[img_id]
                img_w, img_h = image_info['width'], image_info['height']
                img_name = os.path.splitext(image_info['file_name'])[0]

                yolo_lines = []
                for ann in anns:
                    cat_name = self.categories[ann['category_id']]
                    cls_id = self.class_map[cat_name]
                    bb = self.convert_bbox(ann['bbox'], img_w, img_h)
                    yolo_lines.append(f"{cls_id} {' '.join(map(str, bb))}")

                # Salvar arquivo YOLO
                txt_path = os.path.join(self.output_dir, img_name + ".txt")
                with open(txt_path, "w") as f:
                    f.write("\n".join(yolo_lines))

            print(f"Conversão concluída! Arquivos YOLO salvos em: {self.output_dir}")

    """
    converter = COCOtoYOLOConverter(
        json_path="caminho/para/instances_train.json",
        output_dir="caminho/para/saida_yolo"
    )
    converter.convert()

    """
    class YOLOtoCOCOConverter:
        """
        Classe responsável por converter anotações no formato YOLO (.txt)
        para o formato COCO (.json).
        """

        def __init__(self, images_dir, labels_dir, class_names, output_json):
            """
            Inicializa o conversor.

            Args:
                images_dir (str): Pasta contendo as imagens (.jpg, .png, etc.)
                labels_dir (str): Pasta contendo os arquivos YOLO (.txt)
                class_names (list): Lista com os nomes das classes (na ordem dos IDs YOLO)
                output_json (str): Caminho do arquivo de saída COCO (.json)
            """
            self.images_dir = images_dir
            self.labels_dir = labels_dir
            self.class_names = class_names
            self.output_json = output_json

        def convert_bbox(self, bbox, img_width, img_height):
            """
            Converte uma bounding box YOLO para COCO.

            Args:
                bbox (list): [x_center, y_center, width, height] (normalizados)
                img_width (int): Largura da imagem
                img_height (int): Altura da imagem

            Returns:
                list: [x_min, y_min, width, height] (em pixels absolutos)
            """
            x_center, y_center, w, h = bbox
            x_min = (x_center - w / 2) * img_width
            y_min = (y_center - h / 2) * img_height
            width = w * img_width
            height = h * img_height
            return [x_min, y_min, width, height]

        def convert(self):
            """
            Converte todos os arquivos YOLO da pasta para um único JSON COCO.
            """
            images = []
            annotations = []
            categories = []
            ann_id = 1
            img_id = 1

            # Criar categorias COCO
            for i, name in enumerate(self.class_names):
                categories.append({
                    "id": i + 1,
                    "name": name,
                    "supercategory": "object"
                })

            # Percorrer imagens e labels
            for filename in os.listdir(self.labels_dir):
                if not filename.endswith(".txt"):
                    continue

                label_path = os.path.join(self.labels_dir, filename)
                image_name = os.path.splitext(filename)[0]

                # Localiza imagem correspondente
                for ext in [".jpg", ".png", ".jpeg"]:
                    img_path = os.path.join(self.images_dir, image_name + ext)
                    if os.path.exists(img_path):
                        break
                else:
                    print(f"Aviso: imagem não encontrada para {filename}")
                    continue

                # Tamanho da imagem
                with Image.open(img_path) as img:
                    width, height = img.size

                # Adiciona imagem ao JSON
                images.append({
                    "id": img_id,
                    "file_name": os.path.basename(img_path),
                    "width": width,
                    "height": height
                })

                # Lê labels YOLO
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # linha inválida
                    class_id, x, y, w, h = map(float, parts)
                    coco_box = self.convert_bbox([x, y, w, h], width, height)

                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(class_id) + 1,  # COCO começa do 1
                        "bbox": coco_box,
                        "area": coco_box[2] * coco_box[3],
                        "iscrowd": 0
                    })
                    ann_id += 1

                img_id += 1

            # Estrutura final COCO
            coco_output = {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

            # Salva JSON
            with open(self.output_json, "w") as f:
                json.dump(coco_output, f, indent=4)

            print(f"Conversão concluída! Arquivo COCO salvo em: {self.output_json}")

    """
    converter = YOLOtoCOCOConverter(
        images_dir="caminho/para/imagens",
        labels_dir="caminho/para/labels_yolo",
        class_names=["insulator"]
        output_json="saida_coco.json"
    )
    converter.convert()

    """

"""
# EXEMPLO DE USO GERAL

conv = convert()

voc = conv.VOCtoYOLOConverter({'car': 0})
yolo = conv.COCOtoYOLOConverter('coco.json', 'saida/')
coco = conv.YOLOtoCOCOConverter('imagens/', 'labels/', ['car'], 'saida.json')

"""
conv = convert()
coco = conv.YOLOtoCOCOConverter(r"C:\CPLID_val\val\images", r"C:\CPLID_val\val\labels",
                                 ['013'], 'saida.json')

coco.convert()
