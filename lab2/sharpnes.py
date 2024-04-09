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
    QInputDialog,
    QGraphicsBlurEffect,
    QMessageBox,
)
from PyQt5.QtGui import QPixmap, QImage, QColor


class ImageChromaticityApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Sharpness App")
        self.image_label = QLabel(self)
        self.result_label = QLabel(self)
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)

        self.log_button = QPushButton("Нерезкое маскирование", self)
        self.log_button.clicked.connect(self.unsharp_masking)

        self.calculate_sharpness_button = QPushButton("Расчитать резкость")
        self.calculate_sharpness_button.clicked.connect(self.calculate_sharpness)

        self.reset_btn = QPushButton("Сброс", self)
        self.reset_btn.clicked.connect(
            lambda: self.display_image(self.original_image, self.image_label)
        )

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.load_button)
        self.layout.addWidget(self.log_button)
        self.layout.addWidget(self.calculate_sharpness_button)
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
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.original_image = self.image
            self.display_image(self.image, self.image_label)
            self.log_button.setVisible(True)

    def display_image(self, image, label_widget):
        height, width = image.shape[:2]
        bytes_per_line = width
        q_img = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap(q_img)
        label_widget.setPixmap(pixmap)

    def unsharp_masking(self, radius=1, threshold=0, k=1, lambda_val=1):
        radius = 1
        threshold = 1

        k, ok1 = QInputDialog.getDouble(self, "k", "Enter k", value=1.0)
        lambda_val, ok2 = QInputDialog.getDouble(self, "lambda", "Enter lambda", value=1.0)

        if ok1 and ok2:
            # Применение размытия к исходному изображению
            blur_effect = QGraphicsBlurEffect()
            blur_effect.setBlurRadius((2 * k) + 1)
            self.image_label.setGraphicsEffect(blur_effect)

            height, width = self.original_image.shape[:2]
            bytes_per_line = width
            q_img = QImage(
                self.original_image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_Grayscale8,
            )

            img = q_img
            blurred_img = self.image_label.pixmap().toImage()

            for x in range(img.height()):
                for y in range(img.width()):
                    pixel_intensity = QColor(img.pixel(x, y)).lightness()
                    blurred_pixel_intensity = QColor(
                        blurred_img.pixel(x, y)
                    ).lightness()

                    new_intensity = pixel_intensity + (
                        lambda_val * (pixel_intensity - blurred_pixel_intensity)
                    )

                    new_intensity = int(min(max(0, new_intensity), 255))

                    if abs(pixel_intensity - blurred_pixel_intensity) >= threshold:
                        img.setPixel(x, y, new_intensity)

            new_pixmap = QPixmap.fromImage(img)
            self.image_label.setPixmap(new_pixmap)

    def calculate_sharpness(self):
        # Преобразование изображения в массив numpy
        img_array = np.array(self.image)

        # Вычисление градиента по оси X и Y
        gradient_x = np.gradient(img_array, axis=0)
        gradient_y = np.gradient(img_array, axis=1)

        # Вычисление общего градиента как суммы модулей градиентов по осям X и Y
        total_gradient = np.sqrt(gradient_x**2 + gradient_y**2)

        # Среднее значение градиента как метрика резкости
        sharpness_score = np.mean(total_gradient)

        msg = QMessageBox()
        msg.setWindowTitle("Sharpness")
        msg.setText(f"Sharpness: {sharpness_score}")

        msg.exec_()


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
