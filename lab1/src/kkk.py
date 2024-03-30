import os
import random
import sys
import math
import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPixmap, QImage, QTransform
from PyQt5.QtWidgets import (
    QLabel,
    QApplication,
    QMainWindow,
    QInputDialog,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QDialog,
    QRadioButton,
    QPushButton,
    QFileDialog,
    QFrame,
)

os.environ["XDG_SESSION_TYPE"] = "xcb"


class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.image = QPixmap(image_path)
        self.label.setPixmap(self.image)
        self.original_image = self.image

        self.squareFrame = SquareFrame(self)
        self.squareFrame.hide()

        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.mouseMoveEvent

        self.statusBar().showMessage("Hover over the image to see pixel coordinates")
        self.image_path = image_path
        self.average_brightness = None
        self.num_pixels = 0

        self.show_pixel_info_window(
            {
                "x": 0,
                "y": 0,
                "rgb": "RGB: (0, 0, 0)",
                "brightness": 0,
                "average_brightness": self.average_brightness,
                "variance": 0,
                "standard_deviation": 0,
            }
        )

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image.save(file_path)

    def show_input_dialog(self):
        img = cv2.imread(self.image_path, 0)

        dialog = ContrastFormulaDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            contrast_formula = dialog.get_selected_formula()
            if contrast_formula == "окно":
                window_size, window_ok = QInputDialog.getInt(
                    self,
                    "Размер окна",
                    "Введите размер окна для расчета контрастности:",
                )
                if window_ok:
                    contrast_map = self.contrast_custom_window(img, window_size)
            else:
                contrast_map = None
                if contrast_formula == "4 соседа":
                    contrast_map = self.contrast_4_neighbors(img)
                elif contrast_formula == "8 соседей":
                    contrast_map = self.contrast_8_neighbors(img)

            if contrast_map is not None:
                plt.figure()
                plt.imshow(contrast_map, cmap="gray")
                plt.colorbar()
                plt.show()
            else:
                QMessageBox.warning(
                    self, "Ошибка", "Не удалось вычислить контрастную карту"
                )

    def contrast_4_neighbors(self, img):
        contrast_map = np.zeros_like(img, dtype=np.float32)
        for y in range(1, img.shape[0] - 1):
            for x in range(1, img.shape[1] - 1):
                contrast_map[y, x] = (
                    np.abs(int(img[y, x + 1]) - int(img[y, x]))
                    + np.abs(int(img[y, x - 1]) - int(img[y, x]))
                    + np.abs(int(img[y + 1, x]) - int(img[y, x]))
                    + np.abs(int(img[y - 1, x]) - int(img[y, x]))
                )
        return contrast_map

    # Функция для вычисления контрастности по 8 соседям
    def contrast_8_neighbors(self, img):
        contrast_map = np.zeros_like(img, dtype=np.float32)
        for y in range(1, img.shape[0] - 1):
            for x in range(1, img.shape[1] - 1):
                contrast_map[y, x] = (
                    np.abs(int(img[y, x + 1]) - int(img[y, x]))
                    + np.abs(int(img[y, x - 1]) - int(img[y, x]))
                    + np.abs(int(img[y + 1, x]) - int(img[y, x]))
                    + np.abs(int(img[y - 1, x]) - int(img[y, x]))
                    + np.abs(int(img[y + 1, x + 1]) - int(img[y, x]))
                    + np.abs(int(img[y - 1, x - 1]) - int(img[y, x]))
                    + np.abs(int(img[y + 1, x - 1]) - int(img[y, x]))
                    + np.abs(int(img[y - 1, x + 1]) - int(img[y, x]))
                )
        return contrast_map

    # Функция для вычисления контрастности по окну с задаваемым размером
    def contrast_custom_window(self, img, window_size):
        contrast_map = np.zeros_like(img, dtype=np.float32)
        half_size = window_size // 2
        for y in range(half_size, img.shape[0] - half_size):
            for x in range(half_size, img.shape[1] - half_size):
                window = img[
                    y - half_size : y + half_size + 1, x - half_size : x + half_size + 1
                ]
                contrast_map[y, x] = np.std(window)
        return contrast_map

    def decrease_intensity(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)
        channel_factor = 0.7

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = min(255, int(red * channel_factor))
                new_green = min(255, int(green * channel_factor))
                new_blue = min(255, int(blue * channel_factor))
                new_color = QColor(new_red, new_green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def increase_intensity(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)
        channel_factor = 1.3

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = min(255, int(red * channel_factor))
                new_green = min(255, int(green * channel_factor))
                new_blue = min(255, int(blue * channel_factor))
                new_color = QColor(new_red, new_green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def decrease_contrast(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        contrast_level = 0.7
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = ((red / 255 - 0.5) * contrast_level + 0.5) * 255
                new_green = ((green / 255 - 0.5) * contrast_level + 0.5) * 255
                new_blue = ((blue / 255 - 0.5) * contrast_level + 0.5) * 255
                new_red = min(255, max(0, int(new_red)))
                new_green = min(255, max(0, int(new_green)))
                new_blue = min(255, max(0, int(new_blue)))
                new_color = QColor(new_red, new_green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def increase_contrast(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        contrast_level = 1.3
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = ((red / 255 - 0.5) * contrast_level + 0.5) * 255
                new_green = ((green / 255 - 0.5) * contrast_level + 0.5) * 255
                new_blue = ((blue / 255 - 0.5) * contrast_level + 0.5) * 255
                new_red = min(255, max(0, int(new_red)))
                new_green = min(255, max(0, int(new_green)))
                new_blue = min(255, max(0, int(new_blue)))
                new_color = QColor(new_red, new_green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def invert_brightness(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                inverted_red = 255 - red
                inverted_green = 255 - green
                inverted_blue = 255 - blue
                new_color = QColor(inverted_red, inverted_green, inverted_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def apply_random_filter(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = min(
                    255,
                    int(
                        random.uniform(0.0, 1.0) * red
                        + random.uniform(0.0, 1.0) * green
                        + random.uniform(0.0, 1.0) * blue
                    ),
                )
                new_green = min(
                    255,
                    int(
                        random.uniform(0.0, 1.0) * red
                        + random.uniform(0.0, 1.0) * green
                        + random.uniform(0.0, 1.0) * blue
                    ),
                )
                new_blue = min(
                    255,
                    int(
                        random.uniform(0.0, 1.0) * red
                        + random.uniform(0.0, 1.0) * green
                        + random.uniform(0.0, 1.0) * blue
                    ),
                )
                new_color = QColor(new_red, new_green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap.fromImage(new_image)
        self.label.setPixmap(self.image)

    def reset_to_original(self):
        self.image = QPixmap(self.original_image)
        self.label.setPixmap(self.image)

    def get_pixel_info(self, x, y):
        if not self.image.rect().contains(x, y):
            return None

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

        return {
            "x": x,
            "y": y,
            "rgb": rgb,
            "brightness": brightness,
            "average_brightness": self.average_brightness,
            "variance": variance,
            "standard_deviation": standard_deviation,
        }

    def show_pixel_info_window(self, pixel_info):
        if not hasattr(self, "info_window"):
            self.info_window = QWidget()
            self.info_window.setWindowTitle("Pixel Info")
            layout = QVBoxLayout()

            self.info_label = QLabel()
            layout.addWidget(self.info_label)

            button_layout = QGridLayout()

            buttons_data = [
                ("Уменьшение интенсивности", self.decrease_intensity),
                ("Увеличение интенсивности", self.increase_intensity),
                ("Уменьшение контрастности", self.decrease_contrast),
                ("Увеличение контрастности", self.increase_contrast),
                ("Получение негатива", self.invert_brightness),
                ("Обмен цветовых каналов",self.swap_color_channels ),
                ("Симметричное отображение",self.mirror_image ),
                ("Удаление шума", self.denoise_image),
                ("Рандом фильтр", self.apply_random_filter),
                ("Расчёт контрастности", self.show_input_dialog),
                ("Сброс", self.reset_to_original),
                ("Сохранить изображение", self.save_image),
                ("Уменьшение яркости", self.decrease_intensity),
                ("Увеличение яркости", self.increase_intensity),
                ("Уменьшение красного", self.decrease_red),
                ("Увеличение красного", self.increase_red),
                ("Уменьшение зеленого", self.decrease_green),
                ("Увеличение зеленого", self.increase_green),
                ("Уменьшение синего", self.decrease_blue),
                ("Увеличение синего", self.increase_blue),
            ]

            for i, (button_text, button_function) in enumerate(buttons_data):
                button = QPushButton(button_text)
                button.clicked.connect(button_function)
                button_layout.addWidget(button, i // 2, i % 2)

            layout.addLayout(button_layout)
            self.info_window.setLayout(layout)
            self.info_window.setGeometry(100, 100, 600, 150)

        text = (
            f"Координаты пикселя: ({pixel_info['x']}, {pixel_info['y']}) | "
            f"Цвет: {pixel_info['rgb']} | "
            f"Интенсивность: {pixel_info['brightness']} | "
            f"Среднее значение яркости: {pixel_info['average_brightness']} | "
            f"Дисперсия: {pixel_info['variance']} | "
            f"Стандартное отклонение: {pixel_info['standard_deviation']}"
        )

        self.info_label.setText(text)
        self.info_window.show()

    def mirror_image(self):
        # Преобразуем QPixmap в QImage
        qimage = self.image.toImage()

        # Создаем матрицу трансформации для зеркального отражения по горизонтали
        transform = QTransform()
        transform.scale(-1, 1)

        # Применяем трансформацию к изображению
        mirrored_image = qimage.transformed(transform)

        self.image = QPixmap.fromImage(mirrored_image)
        self.label.setPixmap(self.image)

    def denoise_image(self, kernel_size=3):
        # Преобразуем QPixmap в QImage
        qimage = self.image.toImage()

        # Применяем медианный фильтр к каждому пикселю
        for i in range(1, qimage.height() - 1):
            for j in range(1, qimage.width() - 1):
                red_values, green_values, blue_values = [], [], []
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        color = qimage.pixelColor(j + l, i + k)
                        red_values.append(color.red())
                        green_values.append(color.green())
                        blue_values.append(color.blue())
                median_red = sorted(red_values)[len(red_values) // 2]
                median_green = sorted(green_values)[len(green_values) // 2]
                median_blue = sorted(blue_values)[len(blue_values) // 2]
                qimage.setPixelColor(j, i, QColor(median_red, median_green, median_blue))

        self.image = QPixmap.fromImage(qimage)
        self.label.setPixmap(self.image)

    def swap_color_channels(self):
        # Преобразуем QPixmap в QImage
        qimage = self.image.toImage()

        # Обмениваем каналы
        for i in range(qimage.height()):
            for j in range(qimage.width()):
                color = qimage.pixelColor(j, i)
                qimage.setPixelColor(j, i, QColor(color.blue(), color.green(), color.red()))

        self.image = QPixmap.fromImage(qimage)
        self.label.setPixmap(self.image)


    def increase_red(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = min(red + 50, 255)
                new_color = QColor(new_red, green, blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def decrease_red(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_red = max(red - 50, 0)
                new_color = QColor(new_red, green, blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def increase_green(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_green = min(green + 50, 255)
                new_color = QColor(red, new_green, blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def decrease_green(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_green = max(green - 50, 0)
                new_color = QColor(red, new_green, blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def increase_blue(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_blue = min(blue + 50, 255)
                new_color = QColor(red, green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def decrease_blue(self):
        image_qimage = self.image.toImage()
        width, height = image_qimage.width(), image_qimage.height()
        new_image = QImage(width, height, QImage.Format_ARGB32)

        for y in range(height):
            for x in range(width):
                pixel = image_qimage.pixel(x, y)
                red, green, blue, alpha = QColor(pixel).getRgb()
                new_blue = max(blue - 50, 0)
                new_color = QColor(red, green, new_blue, alpha)
                new_image.setPixel(x, y, new_color.rgb())

        self.image = QPixmap(new_image)
        self.label.setPixmap(self.image)

    def mouseMoveEvent(self, event):
        # Получаем координаты курсора
        x = event.x()
        y = event.y()

        self.squareFrame.updatePosition(
            x, y, self.image
        )  # Обновляем положение квадратного фрейма
        self.squareFrame.show()  # Показываем квадратный фрейм

        self.average_brightness, self.num_pixels = (
            self.squareFrame.getAverageBrightnessAndNumPixels()
        )

        self.squareFrame.updatePosition(
            x, y, self.image
        )  # Обновляем положение квадратного фрейма
        self.squareFrame.show()  # Показываем квадратный фрейм
        self.average_brightness, self.num_pixels = (
            self.squareFrame.getAverageBrightnessAndNumPixels()
        )

        # Получаем информацию о пикселе
        pixel_info = self.get_pixel_info(x, y)

        if pixel_info:
            self.show_pixel_info_window(pixel_info)

    def resizeEvent(self, event):
        self.squareFrame.hide()  # Скрываем квадратный фрейм при изменении размера окна
        QMainWindow.resizeEvent(self, event)


class SquareFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(13, 13)
        self.setStyleSheet("background-color: transparent; border: 1px solid yellow;")
        self.image = None
        self.average_brightness = None

    def updatePosition(self, x, y, image):
        self.move(x - 6, y - 6)
        self.image = image
        self.calculateAverageBrightness(x, y, 13, 13)

    def calculateAverageBrightness(self, x, y, size_x, size_y):
        if self.image is None:
            self.average_brightness = None
            return

        total_brightness = 0
        self.num_pixels = 0
        for i in range(x, x + size_x):
            for j in range(y, y + size_y):
                if 0 <= i < self.image.width() and 0 <= j < self.image.height():
                    pixel_color = QColor(self.image.toImage().pixel(i, j))
                    total_brightness += (
                        pixel_color.red() + pixel_color.green() + pixel_color.blue()
                    ) / 3
                    self.num_pixels += 1
        self.average_brightness = int(total_brightness / self.num_pixels)

    def getAverageBrightnessAndNumPixels(self):
        return self.average_brightness, self.num_pixels


class ContrastFormulaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Формула контрастности")

        layout = QVBoxLayout()

        self.four_neighbors_radio = QRadioButton("4 соседа")
        self.eight_neighbors_radio = QRadioButton("8 соседей")
        self.window_radio = QRadioButton("Окно")

        layout.addWidget(self.four_neighbors_radio)
        layout.addWidget(self.eight_neighbors_radio)
        layout.addWidget(self.window_radio)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_selected_formula(self):
        if self.four_neighbors_radio.isChecked():
            return "4 соседа"
        elif self.eight_neighbors_radio.isChecked():
            return "8 соседей"
        elif self.window_radio.isChecked():
            return "окно"
        else:
            return None


import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def show_histogram(img_path):
    img = Image.open(img_path)
    bw = img.convert("L")

    # Создание ч/б представления изображения
    fig1, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(
        2, 4, figsize=(16, 10)
    )
    ax0.imshow(bw, cmap="gray")
    ax0.set_title("Ч/Б представление изображения")

    red_data = [r for r, _, _ in img.getdata()]
    green_data = [g for _, g, _ in img.getdata()]
    blue_data = [b for _, _, b in img.getdata()]

    red_channel_bw = Image.new("L", img.size)
    red_channel_bw.putdata(red_data)
    ax1.imshow(red_channel_bw, cmap="gray")
    ax1.set_title("Красный канал")

    green_channel_bw = Image.new("L", img.size)
    green_channel_bw.putdata(green_data)
    ax2.imshow(green_channel_bw, cmap="gray")
    ax2.set_title("Зеленый канал")

    blue_channel_bw = Image.new("L", img.size)
    blue_channel_bw.putdata(blue_data)
    ax3.imshow(blue_channel_bw, cmap="gray")
    ax3.set_title("Синий канал")

    # Создание графика гистограммы

    def plot_histogram(channels, colors, ax):
        for channel, color, axes in zip(channels, colors, ax):
            # hist_data = channel.histogram()
            # hist_data = [0] + hist_data[:]
            axes.hist(
                channel,
                bins=256,
                range=(0, 256),
                histtype="bar",
                color=color,
                label=f"{color} канал",
            )
            axes.set_title("Гистограмма яркости")
            axes.set_xlabel("Яркость")
            axes.set_ylabel("Частота")
            axes.set_xticks(np.arange(0, 256, 20))
            axes.legend()

    plot_histogram(
        [red_data, green_data, blue_data],
        ["red", "green", "blue"],
        [ax4, ax5, ax6],
    )

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Функция для обновления профиля яркости по выбранной строке
    def update_brightness_profile(row_number):
        brightness_profile = image[row_number, :]
        ax7.clear()
        ax7.plot(brightness_profile, color="b")
        ax7.set_title("Профиль яркости выбранной строки")
        ax7.set_xlabel("Пиксели")
        ax7.set_ylabel("Яркость")

    # Ползунок для выбора строки
    ax_row = plt.axes(
        [0.5, 0.01, 0.4, 0.03]
    )  # Измененные координаты и размеры для слайдера
    slider_row = Slider(ax_row, "Строка", 0, image.shape[0] - 1, valinit=0, valstep=1)

    # Обработчик изменения значения ползунка
    def on_slider_change(val):
        row_number = int(slider_row.val)
        update_brightness_profile(row_number)
        fig1.canvas.draw_idle()

    slider_row.on_changed(on_slider_change)

    # Инициализация графика профиля яркости
    update_brightness_profile(0)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=1.2)

    plt.show()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
        file_path, _ = file_dialog.getOpenFileName()

        if file_path:
            window = ImageWindow(file_path)
            window.show()
            show_histogram(file_path)
    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        os.execv(sys.executable, [sys.executable] + sys.argv)
