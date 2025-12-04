# PIBIT 25/26 - Unificação de anotações:

Código feito por Gabriel Lima Dantas, combase em um código forncido pelo colega de laboratório Lucas Mascarenhas. Solução feita com base nas solicitações do líder de linha de pesquisa Gustavo Rodrigues.

Estrutura de pastas:

/
│
├── voc_to_yolo.py
├── coco_to_yolo.py
├── yolo_to_coco.py
└── convert.py


Exemplo de uso:
    
    Caminho base pode ser relativo ou absoluto
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
