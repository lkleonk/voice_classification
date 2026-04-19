import io

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class TuningReportPDF:
    def __init__(self, title="Hyperparameter tuning with optuna", page_size=A4, margin=inch):
        self.title = title
        self.page_size = page_size
        self.metadata_title = "Evaluation report"
        self.margin = margin
        self.search_space = None
        self.best_params = None
        self.train_config = None
        self.title_for_images = "Training on best params"
        self.classification_report_str = None
        self.clas_rep_title = "Metrics"
        self.benchmark_title = "Benchmark"
        self.benchmark_lines = None
        self.benchmark_lines_old = [
            "For dataset 2025-06-19, the following benchmark is set (Wav2Vec+Parselmouth approach):",
            "./train__norm_off__wav2vec_full/classification_report.txt",
            "Classification Report for Best Model: SVC",
            "Mean Balanced Accuracy: 0.7125",
            "============================================================",
            "              precision    recall  f1-score   support",
            " ",
            "     control       0.41      0.65      0.51        26",
            "        copd       0.89      0.76      0.82        99",
            " ",
            "    accuracy                           0.74       125",
            "   macro avg       0.65      0.71      0.66       125",
            "weighted avg       0.79      0.74      0.75       125",
        ]
        self.final_images = []

    def search_space_to_string(self, search_space_def):
        return "\n".join(search_space_def)

    def split_line_by_max_count(self, text, max_count=66):
        lines = []
        for line in text.split("\n"):
            while len(line) > max_count:
                lines.append(line[:max_count])
                line = line[max_count:]
            lines.append(line)
        lines = "\n".join(lines)
        return lines

    def get_brackets_content(self, search_space):
        processed_search_space = []
        for item in search_space:
            if "(" in item and item.endswith(")"):
                processed_item = item[item.find("(") + 1 : -1]
                processed_search_space.append(processed_item)
            else:
                processed_search_space.append(item)
        return processed_search_space

    def add_search_space(self, search_space: dict):
        search_space_list = [f"{k}: {v}" for k, v in search_space.items()]
        search_space_to_str = "\n".join(search_space_list)
        self.search_space = self.split_line_by_max_count(search_space_to_str)

    def add_best_params(self, best_params: dict):
        self.best_params = best_params

    def add_config_dict(self, config: dict):
        self.train_config = config

    def add_final_training_pil_imgs(self, pil_images: list, new_title_for_images: str = ""):
        if new_title_for_images != "":
            self.title_for_images = new_title_for_images
        self.final_images.extend(pil_images)

    def add_classification_report_str(self, classification_report_str: str):
        self.classification_report_str = classification_report_str

    def change_benchmark(self, benchmark_lines: str):
        self.benchmark_lines = benchmark_lines

    def save_full_pdf(self, path: str):
        c = canvas.Canvas(path, pagesize=self.page_size)
        c.setTitle(self.metadata_title)
        width, height = self.page_size
        y_position = height - self.margin

        c.setFont("Helvetica-Bold", 20)
        c.drawString(self.margin, y_position, self.title)
        y_position -= 40

        if self.search_space:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, y_position, "Search Space")
            y_position -= 25
            c.setFont("Courier", 10)

            for line in self.search_space.split("\n"):
                c.drawString(self.margin + 10, y_position, line)
                y_position -= 14
                if y_position < self.margin:
                    c.showPage()
                    y_position = height - self.margin
                    c.setFont("Courier", 10)

            y_position -= 20

        if self.best_params:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, y_position, "Best Parameters")
            y_position -= 25
            c.setFont("Helvetica", 12)
            for key, val in self.best_params.items():
                line = f"{key}: {val}"
                c.drawString(self.margin + 10, y_position, line)
                y_position -= 16
                if y_position < self.margin:
                    c.showPage()
                    y_position = height - self.margin
            y_position -= 20

        if self.final_images:
            c.setFont("Helvetica-Bold", 18)
            c.drawString(self.margin, y_position, self.title_for_images)
            y_position -= 30
            img_max_width = width * 0.6
            for img in self.final_images:
                aspect = img.height / img.width
                img_width = img_max_width
                img_height = img_width * aspect

                if y_position - img_height < self.margin:
                    c.showPage()
                    y_position = height - self.margin

                img_buffer = io.BytesIO()
                if img.mode == "F":
                    img = img.convert("RGB")
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                rl_image = ImageReader(img_buffer)

                c.drawImage(
                    rl_image,
                    self.margin,
                    y_position - img_height,
                    width=img_width,
                    height=img_height,
                )
                y_position -= (img_height + 15)

        if self.train_config:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, y_position, "Configuration parameters")
            y_position -= 25
            c.setFont("Helvetica", 12)
            for key, val in self.train_config.items():
                line = f"{key}: {val}"
                c.drawString(self.margin + 10, y_position, line)
                y_position -= 16
                if y_position < self.margin:
                    c.showPage()
                    y_position = height - self.margin
            y_position -= 20

        if self.classification_report_str:
            c.showPage()
            y_position = height - self.margin

        if self.classification_report_str:
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, y_position, self.clas_rep_title)
            y_position -= 22
            c.setFont("Courier", 10)
            lines = self.classification_report_str.split("\n")
            for line in lines:
                c.drawString(self.margin + 10, y_position, line)
                y_position -= 12
                if y_position < self.margin:
                    c.showPage()
                    y_position = height - self.margin
                    c.setFont("Courier", 10)
            y_position -= 20

        if self.benchmark_lines and self.benchmark_lines != "":
            c.setFont("Helvetica-Bold", 16)
            c.drawString(self.margin, y_position, self.benchmark_title)
            y_position -= 22
            c.setFont("Courier", 10)
            for line in self.benchmark_lines:
                c.drawString(self.margin + 10, y_position, line)
                y_position -= 12
                if y_position < self.margin:
                    c.showPage()
                    y_position = height - self.margin
                    c.setFont("Courier", 10)
            y_position -= 20

        c.save()
