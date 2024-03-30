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

    def unsharp_masking(self):
        # Запрос параметров у пользователя
        k, ok1 = QInputDialog.getDouble(self, "k", "Enter k", value=1.0)
        lambda_val, ok2 = QInputDialog.getDouble(self, "lambda", "Enter lambda", value=1.0)

        if ok1 and ok2:
            # Применение размытия к исходному изображению
            blur_effect = QGraphicsBlurEffect()
            blur_effect.setBlurRadius(1)  # Можно задать другой радиус размытия
            self.image_label.setGraphicsEffect(blur_effect)

            # Создание копии изображения для обработки
            pixmap = self.image_label.pixmap()
            img = pixmap.toImage()

            # Применение фильтра нерезкого маскирования
            for x in range(img.width()):
                for y in range(img.height()):
                    pixel_color = QColor(img.pixel(x, y))
                    blurred_pixel_color = QColor(pixmap.toImage().pixel(x, y))

                    new_intensity = (
                        pixel_color.lightness()
                        + lambda_val
                        * (pixel_color.lightness() - blurred_pixel_color.lightness())
                        * k
                    )
                    new_intensity = int(min(max(0, new_intensity), 255))

                    new_color = QColor.fromHsl(
                        pixel_color.hue(), pixel_color.saturation(), new_intensity
                    )
                    img.setPixelColor(x, y, new_color)

            # Обновление изображения на метке
            self.image_label.setPixmap(QPixmap(img))

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
