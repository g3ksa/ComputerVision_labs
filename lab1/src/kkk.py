import os
import PIL.Image
import sys
from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor
import math
from PyQt5.QtWidgets import QFileDialog, QPushButton

os.environ["XDG_SESSION_TYPE"] = "xcb"


class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.image = QPixmap(image_path)
        self.label.setPixmap(self.image)

        self.original_image = self.image

        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.mouseMoveEvent

        self.statusBar().showMessage("Hover over the image to see pixel coordinates")

        # Среднее значение яркости
        total_brightness = 0
        self.num_pixels = 0
        for i in range(self.image.width()):
            for j in range(self.image.height()):
                pixel_color = QColor(self.image.toImage().pixel(i, j))
                total_brightness += int(
                    (pixel_color.red() + pixel_color.green() + pixel_color.blue()) / 3
                )
                self.num_pixels += 1
        self.average_brightness = int(total_brightness / self.num_pixels)

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image.save(file_path)

    def decrease_intensity(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()

            new_image = QImage(width, height, QImage.Format_ARGB32)

            channel_factor = 0.7

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Уменьшаем интенсивность каждого цветового канала
                    new_red = min(255, int(red * channel_factor))
                    new_green = min(255, int(green * channel_factor))
                    new_blue = min(255, int(blue * channel_factor))

                    # Создаем новый цвет с уменьшенной интенсивностью и цветовыми каналами
                    new_color = QColor(new_red, new_green, new_blue, alpha)

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def increase_intensity(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()

            new_image = QImage(width, height, QImage.Format_ARGB32)

            channel_factor = 1.3

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Увеличиваем интенсивность каждого цветового канала
                    new_red = min(255, int(red * channel_factor))
                    new_green = min(255, int(green * channel_factor))
                    new_blue = min(255, int(blue * channel_factor))

                    # Создаем новый цвет с увеличенной интенсивностью и цветовыми каналами
                    new_color = QColor(new_red, new_green, new_blue, alpha)

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def decrease_contrast(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()
            contrast_level = 0.7
            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Применяем формулу для уменьшения контрастности
                    new_red = ((red / 255 - 0.5) * contrast_level + 0.5) * 255
                    new_green = ((green / 255 - 0.5) * contrast_level + 0.5) * 255
                    new_blue = ((blue / 255 - 0.5) * contrast_level + 0.5) * 255

                    # Ограничиваем значения каналов
                    new_red = min(255, max(0, int(new_red)))
                    new_green = min(255, max(0, int(new_green)))
                    new_blue = min(255, max(0, int(new_blue)))

                    # Создаем новый цвет с измененной контрастностью
                    new_color = QColor(new_red, new_green, new_blue, alpha)

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def increase_contrast(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()
            contrast_level = 1.3
            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Применяем формулу для уменьшения контрастности
                    new_red = ((red / 255 - 0.5) * contrast_level + 0.5) * 255
                    new_green = ((green / 255 - 0.5) * contrast_level + 0.5) * 255
                    new_blue = ((blue / 255 - 0.5) * contrast_level + 0.5) * 255

                    # Ограничиваем значения каналов
                    new_red = min(255, max(0, int(new_red)))
                    new_green = min(255, max(0, int(new_green)))
                    new_blue = min(255, max(0, int(new_blue)))

                    # Создаем новый цвет с измененной контрастностью
                    new_color = QColor(new_red, new_green, new_blue, alpha)

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def invert_brightness(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()

            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Инвертируем значения цветовых каналов
                    inverted_red = 255 - red
                    inverted_green = 255 - green
                    inverted_blue = 255 - blue

                    # Создаем новый цвет с инвертированной яркостью
                    new_color = QColor(
                        inverted_red, inverted_green, inverted_blue, alpha
                    )

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def apply_sepia_filter(self):
        if hasattr(self, "image") and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()

            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    pixel = image_qimage.pixel(x, y)

                    red, green, blue, alpha = QColor(pixel).getRgb()

                    # Применяем фильтр сепии
                    new_red = min(255, int(0.393 * red + 0.769 * green + 0.189 * blue))
                    new_green = min(
                        255, int(0.349 * red + 0.686 * green + 0.168 * blue)
                    )
                    new_blue = min(255, int(0.272 * red + 0.534 * green + 0.131 * blue))

                    # Создаем новый цвет с эффектом сепии
                    new_color = QColor(new_red, new_green, new_blue, alpha)

                    # Устанавливаем новый цвет для пикселя
                    new_image.setPixel(x, y, new_color.rgb())

            # Обновляем изображение
            self.image = QPixmap.fromImage(new_image)
            self.label.setPixmap(self.image)

    def reset_to_original(self):
        self.image = QPixmap(self.original_image)
        self.label.setPixmap(self.image)

    def mouseMoveEvent(self, event):
        if not self.image.rect().contains(event.pos()):
            return

        x = event.x()
        y = event.y()

        pixel_color = QColor(self.image.toImage().pixel(x, y))
        rgb = f"RGB: ({pixel_color.red()}, {pixel_color.green()}, {pixel_color.blue()})"
        brightness = int(
            (pixel_color.red() + pixel_color.green() + pixel_color.blue()) / 3
        )

        variance = 0
        for i in range(self.image.width()):
            for j in range(self.image.height()):
                if i == x and j == y:
                    pixel_color = QColor(self.image.toImage().pixel(i, j))
                    brightness1 = int(
                        (pixel_color.red() + pixel_color.green() + pixel_color.blue())
                        / 3
                    )
                    variance += (brightness1 - self.average_brightness) ** 2

        variance /= self.num_pixels
        standard_deviation = math.sqrt(variance)

        # self.statusBar().showMessage(f"Координаты пикселя, цвет и интенсивность: ({x}, {y}) | {rgb} | {brightness}")
        # Создаем текст для отображения в окне с информацией о пикселе

        text = f"Координаты пикселя: ({x}, {y}) | Цвет: {rgb} | Интенсивность: {brightness} | Среднее значение яркости: {self.average_brightness} | Дисперсия: {variance} | Стандартное отклонение: {standard_deviation}"
        # text = f"Координаты пикселя: ({x}, {y}) | Цвет: {rgb} | Интенсивность: {brightness} | Среднее значение яркости: {}"

        # Проверяем, существует ли уже окно с информацией о пикселе
        if not hasattr(self, "info_window"):
            self.info_window = QWidget()
            self.info_window.setWindowTitle("Pixel Info")
            layout = QVBoxLayout()

            self.info_label = QLabel()
            layout.addWidget(self.info_label)

            button_layout = QHBoxLayout()

            buttons_data = [
                ("Сохранить изображение", self.save_image),
                ("Увелечение интенсивности", self.increase_intensity),
                ("Уменьшение интенсивности", self.decrease_intensity),
                ("Повышение контрастности", self.increase_contrast),
                ("Уменьшение контрастности", self.decrease_contrast),
                ("Получение негатива", self.invert_brightness),
                ("Обмен цветовых каналов", self.save_image),
                ("Симметрочное отображение", self.save_image),
                ("Удаление шума", self.save_image),
                ("Сепия фильтр", self.apply_sepia_filter),
                ("Сброс", self.reset_to_original),
            ]

            for button_text, button_function in buttons_data:
                button = QPushButton(button_text)
                button.setFixedSize(150, 50)
                button.clicked.connect(button_function)
                button_layout.addWidget(button)

            layout.addLayout(button_layout)

            self.info_window.setLayout(layout)
            self.info_window.setGeometry(100, 100, 600, 150)
            self.info_window.show()
        # Обновляем текст в окне с информацией о пикселе
        self.info_label.setText(text)


import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow("1.png")
    window.show()

    # Загрузка изображения
    image = cv2.imread("1.png", cv2.IMREAD_GRAYSCALE)

    # Создание графического окна
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Инициализация начальной строки для анализа (в данном случае - центральная строка)
    initial_row_number = image.shape[0] // 2

    # Функция для обновления профиля яркости по выбранной строке
    def update_brightness_profile(row_number):
        brightness_profile = image[row_number, :]
        ax.clear()
        ax.plot(brightness_profile, color="b")
        ax.set_title("Яркостный профиль выбранной строки")
        ax.set_xlabel("Пиксели")
        ax.set_ylabel("Яркость")
        fig.canvas.draw_idle()

    # Ползунок для выбора строки
    ax_row = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider_row = Slider(
        ax_row, "Строка", 0, image.shape[0] - 1, valinit=initial_row_number, valstep=1
    )

    # Обработчик изменения значения ползунка
    def on_slider_change(val):
        row_number = int(slider_row.val)
        update_brightness_profile(row_number)

    slider_row.on_changed(on_slider_change)

    # Инициализация графика профиля яркости
    update_brightness_profile(initial_row_number)

    plt.show()
