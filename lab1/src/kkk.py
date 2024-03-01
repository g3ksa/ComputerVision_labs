import os
import random
import sys
from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QInputDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
import math
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QFileDialog, QPushButton, QFrame
import numpy as np
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5.QtGui import QColor, QPainter, QPen, QPolygon
from PyQt5.QtWidgets import QDialog, QRadioButton, QVBoxLayout, QPushButton

os.environ["XDG_SESSION_TYPE"] = "xcb"


class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super().__init__()

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.image = QPixmap(image_path)
        self.label.setPixmap(self.image)

        self.squareFrame = SquareFrame(self)
        self.squareFrame.hide()

        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.mouseMoveEvent

        self.statusBar().showMessage("Hover over the image to see pixel coordinates")
        self.alt_pressed = False

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
            
    def show_input_dialog(self):
        img = cv2.imread("1.png", 0)

        dialog = ContrastFormulaDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            contrast_formula = dialog.get_selected_formula()
            if contrast_formula == "окно":
                window_size, window_ok = QInputDialog.getInt(self, "Размер окна", "Введите размер окна для расчета контрастности:")
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
                QMessageBox.warning(self, "Ошибка", "Не удалось вычислить контрастную карту")
            
        
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
                ("Сохранить изображение", self.save_image),
                ("Увеличение интенсивности", self.increase_intensity),
                ("Повышение контрастности", self.increase_contrast),
                ("Получение негатива", self.invert_brightness),
                ("Уменьшение интенсивности", self.decrease_intensity),
                ("Обмен цветовых каналов", self.save_image),
                ("Уменьшение контрастности", self.decrease_contrast),
                ("Симметричное отображение", self.save_image),
                ("Удаление шума", self.save_image),
                ("Ретро фильтр", self.apply_random_filter),
                ("Сброс", self.reset_to_original),
                ("Расчёт контрастности",self.show_input_dialog),
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

    def mouseMoveEvent(self, event):
        # Получаем координаты курсора
        x = event.x()
        y = event.y()
        self.alt_pressed = event.modifiers() & Qt.AltModifier

        # Получаем информацию о пикселе
        pixel_info = self.get_pixel_info(x, y)

        if pixel_info:
            # Показываем окно с информацией о пикселе
            self.show_pixel_info_window(pixel_info)

        self.squareFrame.updatePosition(x, y)  # Обновляем положение квадратного фрейма
        self.squareFrame.show()  # Показываем квадратный фрейм

    def resizeEvent(self, event):
        self.squareFrame.hide()  # Скрываем квадратный фрейм при изменении размера окна
        QMainWindow.resizeEvent(self, event)


class SquareFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(13, 13)
        self.setStyleSheet("background-color: transparent; border: 1px solid yellow;")

    def updatePosition(self, x, y):
        self.move(x - 6, y - 6)

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
