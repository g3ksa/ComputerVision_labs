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
        self.setWindowTitle("Image Sharpness App")
        self.image_label = QLabel(self)
        self.result_label = QLabel(self)
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)

        self.log_button = QPushButton("Нерезкое маскирование", self)
        self.log_button.clicked.connect(lambda: self.apply_filter(self.unsharp_masking))

        self.reset_btn = QPushButton("Сброс", self)
        self.reset_btn.clicked.connect(
            lambda: self.display_image(self.original_image, self.image_label)
        )

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.log_button)
        self.layout.addWidget(self.reset_btn)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.result_label)
        self.setLayout(self.layout)
        self.image = None
        self.original_image = None

        self.log_button.setVisible(False)

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
        else:
            self.image = filter_func(self.image)
        self.display_image(self.image, self.image_label)

    def gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)
                / (2 * sigma**2)
            ),
            (size, size),
        )
        return kernel / np.sum(kernel)

    def unsharp_masking(
        self, image, kernel_size=(5, 5), sigma=1.0, alpha=1.5, beta=-0.5
    ):
        blurred = cv2.filter2D(image, -1, self.gaussian_kernel(kernel_size[0], sigma))
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        laplacian = cv2.filter2D(blurred, -1, laplacian_kernel)
        sharpened = image + alpha * laplacian + beta
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return sharpened


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
