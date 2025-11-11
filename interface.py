import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from convert import convert  # importa seu módulo (ajuste o nome do arquivo se necessário)

class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Conversor de Anotações")
        self.root.geometry("450x350")

        tk.Label(root, text="Escolha o tipo de conversão:", font=("Arial", 12, "bold")).pack(pady=10)

        self.mode = tk.StringVar(value="YOLO → COCO")
        tk.Radiobutton(root, text="PASCAL VOC → YOLO", variable=self.mode, value="VOC → YOLO").pack(anchor="w", padx=40)
        tk.Radiobutton(root, text="COCO → YOLO", variable=self.mode, value="COCO → YOLO").pack(anchor="w", padx=40)
        tk.Radiobutton(root, text="YOLO → COCO", variable=self.mode, value="YOLO → COCO").pack(anchor="w", padx=40)

        tk.Button(root, text="Selecionar Arquivos/Pastas", command=self.select_paths, width=25).pack(pady=20)
        tk.Button(root, text="Converter", command=self.run_conversion, width=25, bg="#4CAF50", fg="white").pack(pady=10)

        self.selected_paths = {}

    def select_paths(self):
        mode = self.mode.get()
        paths = {}

        if mode == "VOC → YOLO":
            paths["xml_dir"] = filedialog.askdirectory(title="Selecione a pasta com arquivos XML")
            paths["output_dir"] = filedialog.askdirectory(title="Selecione a pasta de saída")
        elif mode == "COCO → YOLO":
            paths["json_path"] = filedialog.askopenfilename(title="Selecione o arquivo JSON COCO", filetypes=[("JSON files", "*.json")])
            paths["output_dir"] = filedialog.askdirectory(title="Selecione a pasta de saída")
        elif mode == "YOLO → COCO":
            paths["images_dir"] = filedialog.askdirectory(title="Selecione a pasta com imagens")
            paths["labels_dir"] = filedialog.askdirectory(title="Selecione a pasta com labels YOLO")
            paths["output_json"] = filedialog.asksaveasfilename(title="Salvar como", defaultextension=".json", filetypes=[("JSON", "*.json")])

        self.selected_paths = paths
        messagebox.showinfo("Arquivos Selecionados", f"Selecionado para {mode}:\n{paths}")

    def run_conversion(self):
        if not self.selected_paths:
            messagebox.showerror("Erro", "Nenhum arquivo ou pasta selecionado!")
            return

        conv = convert()
        mode = self.mode.get()

        try:
            if mode == "VOC → YOLO":
                class_map = {"classe": 0}  # substitua conforme necessário
                converter = conv.VOCtoYOLOConverter(class_map)
                for xml_file in Path(self.selected_paths["xml_dir"]).glob("*.xml"):
                    yolo_lines = converter.convert_xml(str(xml_file))
                    out_path = Path(self.selected_paths["output_dir"]) / (xml_file.stem + ".txt")
                    with open(out_path, "w") as f:
                        f.write("\n".join(yolo_lines))

            elif mode == "COCO → YOLO":
                converter = conv.COCOtoYOLOConverter(
                    self.selected_paths["json_path"],
                    self.selected_paths["output_dir"]
                )
                converter.convert()

            elif mode == "YOLO → COCO":
                class_names = ["classe"]  # substitua conforme necessário
                converter = conv.YOLOtoCOCOConverter(
                    self.selected_paths["images_dir"],
                    self.selected_paths["labels_dir"],
                    class_names,
                    self.selected_paths["output_json"]
                )
                converter.convert()

            messagebox.showinfo("Sucesso", f"Conversão concluída com sucesso!\nModo: {mode}")

        except Exception as e:
            messagebox.showerror("Erro durante a conversão", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()
