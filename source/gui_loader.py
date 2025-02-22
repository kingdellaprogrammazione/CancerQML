import json
import tkinter as tk
from tkinter import ttk

class MetadataGUI:
    def __init__(self, root, metadata_file):
        self.root = root
        self.metadata_file = metadata_file
        self.metadata = self.load_metadata()
        self.options = self.get_options()
        self.result = []  # initialize the result attribute

        self.create_widgets()

    def load_metadata(self):
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def get_options(self):
        options = {
            "model_type": set(),
            "downsampling": set(),
            "pca": set(),
            "encoding": set(),
            "ansatz": set(),
            "layers": set(),
            "learning_rate": set(),
            "epochs": set(),
            "features": set()
        }

        for file in self.metadata.values():
            options["model_type"].add(file["model_type"])
            options["downsampling"].add(file["downsampling"])
            options["pca"].add(file["pca"])
            options["encoding"].add(file["encoding"])
            options["ansatz"].add(file["ansatz"])
            options["layers"].add(file["layers"])
            options["learning_rate"].add(file["learning_rate"])
            options["epochs"].add(file["epochs"])
            options["features"].add(file["features"])

        return options

    def create_widgets(self):
        self.model_type_label = tk.Label(self.root, text="Model Type:")
        self.model_type_label.grid(row=0, column=0)
        self.model_type_var = tk.StringVar()
        if len(self.options["model_type"]) == 1:
            self.model_type_var.set(list(self.options["model_type"])[0])
            self.model_type_menu = tk.Label(self.root, text=list(self.options["model_type"])[0])
            self.model_type_menu.grid(row=0, column=1)
        else:
            self.model_type_menu = ttk.Combobox(self.root, textvariable=self.model_type_var)
            self.model_type_menu['values'] = list(self.options["model_type"])
            self.model_type_menu.grid(row=0, column=1)

        self.downsampling_label = tk.Label(self.root, text="Downsampling:")
        self.downsampling_label.grid(row=1, column=0)
        self.downsampling_var = tk.BooleanVar()
        self.downsampling_checkbox = tk.Checkbutton(self.root, variable=self.downsampling_var)
        self.downsampling_checkbox.grid(row=1, column=1)

        self.pca_label = tk.Label(self.root, text="PCA:")
        self.pca_label.grid(row=2, column=0)
        self.pca_var = tk.BooleanVar()
        self.pca_checkbox = tk.Checkbutton(self.root, variable=self.pca_var)
        self.pca_checkbox.grid(row=2, column=1)

        self.encoding_label = tk.Label(self.root, text="Encoding:")
        self.encoding_label.grid(row=3, column=0)
        self.encoding_var = tk.StringVar()
        if len(self.options["encoding"]) == 1:
            self.encoding_var.set(list(self.options["encoding"])[0])
            self.encoding_menu = tk.Label(self.root, text=list(self.options["encoding"])[0])
            self.encoding_menu.grid(row=3, column=1)
        else:
            self.encoding_menu = ttk.Combobox(self.root, textvariable=self.encoding_var)
            self.encoding_menu['values'] = list(self.options["encoding"])
            self.encoding_menu.grid(row=3, column=1)
    
        self.ansatz_label = tk.Label(self.root, text="Ansatz:")
        self.ansatz_label.grid(row=4, column=0)
        self.ansatz_var = tk.StringVar()
        if len(self.options["ansatz"]) == 1:
            self.ansatz_var.set(list(self.options["ansatz"])[0])
            self.ansatz_menu = tk.Label(self.root, text=list(self.options["ansatz"])[0])
            self.ansatz_menu.grid(row=4, column=1)
        else:
            self.ansatz_menu = ttk.Combobox(self.root, textvariable=self.ansatz_var)
            self.ansatz_menu['values'] = list(self.options["ansatz"])
            self.ansatz_menu.grid(row=4, column=1)
    
        self.layers_label = tk.Label(self.root, text="Layers:")
        self.layers_label.grid(row=5, column=0)
        self.layers_var = tk.IntVar()
        if len(self.options["layers"]) == 1:
            self.layers_var.set(list(self.options["layers"])[0])
            self.layers_menu = tk.Label(self.root, text=str(list(self.options["layers"])[0]))
            self.layers_menu.grid(row=5, column=1)
        else:
            self.layers_menu = ttk.Combobox(self.root, textvariable=self.layers_var)
            self.layers_menu['values'] = list(self.options["layers"])
            self.layers_menu.grid(row=5, column=1)
    
        self.learning_rate_label = tk.Label(self.root, text="Learning Rate:")
        self.learning_rate_label.grid(row=6, column=0)
        self.learning_rate_var = tk.DoubleVar()
        if len(self.options["learning_rate"]) == 1:
            self.learning_rate_var.set(list(self.options["learning_rate"])[0])
            self.learning_rate_menu = tk.Label(self.root, text=str(list(self.options["learning_rate"])[0]))
            self.learning_rate_menu.grid(row=6, column=1)
        else:
            self.learning_rate_menu = ttk.Combobox(self.root, textvariable=self.learning_rate_var)
            self.learning_rate_menu['values'] = list(self.options["learning_rate"])
            self.learning_rate_menu.grid(row=6, column=1)
    
        self.epochs_label = tk.Label(self.root, text="Epochs:")
        self.epochs_label.grid(row=7, column=0)
        self.epochs_var = tk.IntVar()
        if len(self.options["epochs"]) == 1:
            self.epochs_var.set(list(self.options["epochs"])[0])
            self.epochs_menu = tk.Label(self.root, text=str(list(self.options["epochs"])[0]))
            self.epochs_menu.grid(row=7, column=1)
        else:
            self.epochs_menu = ttk.Combobox(self.root, textvariable=self.epochs_var)
            self.epochs_menu['values'] = list(self.options["epochs"])
            self.epochs_menu.grid(row=7, column=1)
    
        self.features_label = tk.Label(self.root, text="Features:")
        self.features_label.grid(row=8, column=0)
        self.features_var = tk.IntVar()
        if len(self.options["features"]) == 1:
            self.features_var.set(list(self.options["features"])[0])
            self.features_menu = tk.Label(self.root, text=str(list(self.options["features"])[0]))
            self.features_menu.grid(row=8, column=1)
        else:
            self.features_menu = ttk.Combobox(self.root, textvariable=self.features_var)
            self.features_menu['values'] = list(self.options["features"])
            self.features_menu.grid(row=8, column=1)

        self.search_button = tk.Button(self.root, text="Search", command=self.get_result)
        self.search_button.grid(row=9, column=0, columnspan=2)

        self.terminate_button = tk.Button(self.root, text="Terminate", command=self.root.destroy)
        self.terminate_button.grid(row=11, column=0, columnspan=2)

        self.result_label = tk.Label(self.root, text="Result:")
        self.result_label.grid(row=10, column=0)
        self.result_text = tk.Text(self.root, height=10, width=40)
        self.result_text.grid(row=10, column=1)

    def get_result(self):
        model_type = self.model_type_var.get()
        downsampling = self.downsampling_var.get()
        pca = self.pca_var.get()
        encoding = self.encoding_var.get()
        ansatz = self.ansatz_var.get()
        layers = self.layers_var.get()
        learning_rate = self.learning_rate_var.get()
        epochs = self.epochs_var.get()
        features = self.features_var.get()

        result = None
        for file, metadata in self.metadata.items():
            if (metadata["model_type"] == model_type and
                    metadata["downsampling"] == downsampling and
                    metadata["pca"] == pca and
                    metadata["encoding"] == encoding and
                    metadata["ansatz"] == ansatz and
                    metadata["layers"] == layers and
                    metadata["learning_rate"] == learning_rate and
                    metadata["epochs"] == epochs and
                    metadata["features"] == features):
                result = {
                    "model_file": file,
                    "metadata": metadata
                }
                break
        if result:
            self.result.append(result)
            self.result_text.delete('1.0', tk.END)
            for i, res in enumerate(self.result):
                #self.result_text.insert(tk.END, f"Result {i+1}:\nModel file: {res['model_file']}\nMetadata: {res['metadata']}\n\n")
                self.result_text.insert(tk.END, f"Success")
        else:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, "Search failed. No matching model found.")


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Metadata GUI")
    gui = MetadataGUI(root, "metadata.json")
    root.mainloop()
    print(gui.result)