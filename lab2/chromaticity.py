import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QLineEdit,
)
from PyQt5.QtGui import QPixmap, QImage


class ImageChromaticityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Chromaticity App")
        self.image_label = QLabel(self)
        self.result_label = QLabel(self)
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)

        self.log_button = QPushButton("Логарифмическое преобразование", self)
        self.log_button.clicked.connect(
            lambda: self.apply_filter(self.logarithmic_transformation)
        )

        self.bin_button = QPushButton("Бинарное преобразование", self)
        self.bin_button.clicked.connect(
            lambda: self.apply_filter(self.binary_threshold, threshold=128)
        )

        self.power_button = QPushButton("Степенное преобразование", self)
        self.power_button.clicked.connect(self.show_gamma_input)

        self.range_const_button = QPushButton("Вырезание диапозона с константой", self)
        self.range_const_button.clicked.connect(
            lambda: self.apply_filter(
                self.brightness_range_cut,
                min_value=100,
                max_value=200,
                approach="constant",
                constant_value=50,
            )
        )

        self.range_preserve_button = QPushButton(
            "Вырезание диапозона с сохранением", self
        )
        self.range_preserve_button.clicked.connect(
            lambda: self.apply_filter(
                self.brightness_range_cut,
                min_value=100,
                max_value=200,
                approach="preserve",
            )
        )

        self.reset_btn = QPushButton("Сброс", self)
        self.reset_btn.clicked.connect(
            lambda: self.display_image(self.original_image, self.image_label)
        )

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.log_button)
        self.layout.addWidget(self.power_button)
        self.layout.addWidget(self.bin_button)
        self.layout.addWidget(self.range_const_button)
        self.layout.addWidget(self.range_preserve_button)
        self.layout.addWidget(self.reset_btn)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)
        self.image = None
        self.original_image = None

        self.gamma_input = QLineEdit(self)
        self.gamma_input.setPlaceholderText("Введите значение гаммы")
        self.gamma_input.setVisible(False)
        self.layout.addWidget(self.gamma_input)

        self.gamma_input.returnPressed.connect(
            lambda: self.apply_filter(
                self.power_transformation, gamma=float(self.gamma_input.text())
            )
        )

        self.log_button.setVisible(False)
        self.power_button.setVisible(False)
        self.bin_button.setVisible(False)
        self.range_const_button.setVisible(False)
        self.range_preserve_button.setVisible(False)

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            self.original_image = self.image
            self.display_image(self.image, self.image_label)
            self.log_button.setVisible(True)
            self.power_button.setVisible(True)
            self.bin_button.setVisible(True)
            self.range_const_button.setVisible(True)
            self.range_preserve_button.setVisible(True)

    def display_image(self, image, label_widget):
        self.image = image
        height, width = image.shape[:2]
        bytes_per_line = width * image.shape[2]
        q_img = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap(q_img)
        label_widget.setPixmap(pixmap)

    def apply_filter(
        self,
        filter_func,
        gamma=None,
        threshold=None,
        approach=None,
        constant_value=None,
        min_value=None,
        max_value=None,
    ):
        if self.image is None:
            return
        elif gamma:
            self.image = filter_func(self.image, gamma)
        elif threshold:
            self.image = filter_func(self.image, threshold)
        elif approach:
            self.image = filter_func(
                self.image,
                min_value,
                max_value,
                approach=approach,
                constant_value=constant_value,
            )
        else:
            self.image = filter_func(self.image)
        self.display_image(self.image, self.image_label)

    def logarithmic_transformation(self, image):
        max_val = np.max(image)

        palette_range = 256

        c = (palette_range - 1) / np.log(1 + max_val)
        transformed_image = c * np.log(1 + image)

        transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)

        return transformed_image

    def power_transformation(self, image, gamma):
        min_val = np.min(image)
        max_val = np.max(image)
        palette_range = 256
        c = (palette_range - 1) / (max_val**gamma - min_val**gamma)
        transformed_image = c * (image**gamma)
        transformed_image = np.clip(transformed_image, 0, 255).astype(np.uint8)
        self.gamma_input.setVisible(False)
        return transformed_image

    def binary_threshold(self, image, threshold):
        binary_image = np.zeros_like(image, dtype=np.uint8)

        binary_image[image > threshold] = 255

        return binary_image

    def brightness_range_cut(
        self, image, min_val, max_val, approach="constant", constant_value=0
    ):
        processed_image = image.copy()

        if approach == "constant":
            processed_image[image < min_val] = constant_value
            processed_image[image > max_val] = constant_value
        elif approach == "preserve":
            pass

        return processed_image

    def show_gamma_input(self):
        self.gamma_input.setVisible(True)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = ImageChromaticityApp()
        window.setGeometry(100, 100, 800, 600)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QApplication.critical(None, "Error", str(e))
        os.execv(sys.executable, [sys.executable] + sys.argv)
