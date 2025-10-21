# PIBIT 25/26 - Unificação de anotações:

Código feito por Gabriel Lima Dantas, combase em um código forncido pelo colega de laboratório Lucas Mascarenhas. Solução feita com base nas solicitações do líder de linha de pesquisa Gustavo Rodrigues.

Conversor de:

* VOC para YOLO;
* COCO para YOLO;
* YOLO para COCO.

Modo de uso:

# Uso 
   yolo_labels = converter.convert_xml("/arquivo.xml") // caminho

    Salvar em arquivo .txt, talvez seja melhor
    with open("saida.txt", "w") as f:
        f.write("\n".join(yolo_labels))

 converter = COCOtoYOLOConverter(
        json_path="caminho/para/instances_train.json",
        output_dir="caminho/para/saida_yolo"
    )
    converter.convert()

 converter = YOLOtoCOCOConverter(
        images_dir="caminho/para/imagens",
        labels_dir="caminho/para/labels_yolo",
        class_names=["insulator"]
        output_json="saida_coco.json"
    )
    converter.convert()

# EXEMPLO DE USO GERAL

conv = convert()

voc = conv.VOCtoYOLOConverter({'car': 0})
yolo = conv.COCOtoYOLOConverter('coco.json', 'saida/')
coco = conv.YOLOtoCOCOConverter('imagens/', 'labels/', ['car'], 'saida.json')

coco.convert()
