import cv2
import random
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from PIL import Image
import sys
from PyQt5.QtWidgets import QLabel, QApplication, QMainWindow, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog, QLineEdit
from PyQt5.QtGui import QColor
import math
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication, QComboBox
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import QMainWindow, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QColor, QImage, QPainter
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5.QtWidgets import QWidget, QLabel,QFileDialog, QMessageBox
from PyQt5.QtGui import QColor, QPixmap, QImage
from PyQt5.QtWidgets import QGraphicsBlurEffect
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import QLocale



def calculate_sharpness(image):
    sharpness = 0
    for x in range(image.width() - 1):
        for y in range(image.height() - 1):
            pixel_color = QColor(image.pixel(x, y))
            right_pixel_color = QColor(image.pixel(x + 1, y))
            bottom_pixel_color = QColor(image.pixel(x, y + 1))

            diff_right = abs(pixel_color.lightness() - right_pixel_color.lightness())
            diff_bottom = abs(pixel_color.lightness() - bottom_pixel_color.lightness())

            sharpness += diff_right + diff_bottom

    return sharpness

class WindowDialog(QDialog):
    def __init__(self, parent=None):
        super(WindowDialog, self).__init__(parent)
        self.setWindowTitle("Введите параметр")

        self.layout = QVBoxLayout(self)
        
        self.label_k = QLabel("Введите значение параметра:")
        self.edit_k = QLineEdit()
        self.edit_k.setValidator(QDoubleValidator())

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.button_ok)

    def check_and_accept(self):
        k = self.get_parameters()
        if k is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Неверное значение")

    def get_parameters(self):
        k = self.edit_k.text().strip()
        if not k:
            return None
        try:
            k = float(k)
            if k >=0:
                return k
            else:
                return None
        except ValueError:
            return None
        
class WindowDialogAddNoise(QDialog):
    def __init__(self, parent=None):
        super(WindowDialogAddNoise, self).__init__(parent)
        self.setWindowTitle("Введите коэффициент шума")

        self.layout = QVBoxLayout(self)
        
        self.label_k = QLabel("Введите коэффициент шума (от 0 до 1):")
        self.edit_k = QLineEdit()
        validator = QDoubleValidator(0.0, 1.0, 2)
        self.edit_k.setValidator(validator)

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.button_ok)

    def check_and_accept(self):
        k = self.get_parameters()
        if k is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Неверное значение")

    def get_parameters(self):
        k = self.edit_k.text().strip()
        if not k:
            return None
        try:
            k = float(k)
            if 0.0 <= k <= 1.0:
                return k
            else:
                return None
        except ValueError:
            return None

class WindowDialogSigmaForGauss(QDialog):
    def __init__(self, parent=None):
        super(WindowDialogSigmaForGauss, self).__init__(parent)
        self.setWindowTitle("Введите значение для сигма")

        self.layout = QVBoxLayout(self)
        
        self.label_k = QLabel("Введите значения для сигма (от 0.5 до 5):")
        self.edit_k = QLineEdit()
        validator = QDoubleValidator(0.1, 5, 2)
        self.edit_k.setValidator(validator)

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.button_ok)

    def check_and_accept(self):
        k = self.get_parameters()
        if k is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Неверное значение")

    def get_parameters(self):
        k = self.edit_k.text().strip()
        if not k:
            return None
        try:
            k = float(k)
            if 0.5 <= k <= 5.0:
                return k
            else:
                return None
        except ValueError:
            return None
        
class WindowDialogSigma(QDialog):
    def __init__(self, parent=None):
        super(WindowDialogSigma, self).__init__(parent)
        self.setWindowTitle("Введите значения для сигма и k")

        self.layout = QVBoxLayout(self)
        
        self.label_sigma = QLabel("Введите значение для сигма (от 0.5 до 5):")
        self.edit_sigma = QLineEdit()
        validator_sigma = QDoubleValidator(0.5, 5, 2)
        self.edit_sigma.setValidator(validator_sigma)

        self.label_k = QLabel("Введите значение для k (от 1 до 3):")
        self.edit_k = QLineEdit()
        validator_k = QIntValidator(1, 3)
        self.edit_k.setValidator(validator_k)

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_sigma)
        self.layout.addWidget(self.edit_sigma)
        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.button_ok)

    def check_and_accept(self):
        sigma, k = self.get_parameters()
        if sigma is not None and k is not None:
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Неверные значения")

    def get_parameters(self):
        sigma_value = self.edit_sigma.text().strip()
        k_value = self.edit_k.text().strip()
        
        try:
            sigma = float(sigma_value) if sigma_value else None
            k = int(k_value) if k_value else None
        except ValueError:
            return None, None
        
        if sigma is not None and 0.5 <= sigma <= 5.0 and k is not None and 1 <= k <= 3:
            return sigma, k
        else:
            return None, None

class WindowDialogKernel(QDialog):
    def __init__(self, parent=None):
        super(WindowDialogKernel, self).__init__(parent)
        self.setWindowTitle("Выберите размер ядра")

        self.layout = QVBoxLayout(self)
        
        self.label_kernel = QLabel("Выберите размер ядра:")
        self.combo_kernel = QComboBox()
        self.combo_kernel.addItems(["3x3", "5x5"])

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_kernel)
        self.layout.addWidget(self.combo_kernel)
        self.layout.addWidget(self.button_ok)

    def get_parameters(self):
        selected_text = self.combo_kernel.currentText()
        if selected_text == "3x3":
            return 3
        elif selected_text == "5x5":
            return 5

    def check_and_accept(self):
        self.accept()


class ParametersDialog(QDialog):
    def __init__(self, parent=None):
        super(ParametersDialog, self).__init__(parent)
        self.setWindowTitle("Введите параметры k и lambda")

        self.layout = QVBoxLayout(self)
        
        self.label_k = QLabel("Введите значение k:")
        self.edit_k = QLineEdit()
        self.edit_k.setValidator(QDoubleValidator())

        self.label_c = QLabel("Введите значение lambda:")
        self.edit_c = QLineEdit()
        self.edit_c.setValidator(QDoubleValidator())

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.label_c)
        self.layout.addWidget(self.edit_c)
        self.layout.addWidget(self.button_ok)
    
    def check_and_accept(self):
        parameters = self.get_parameters()
        if parameters is not None:
            k, lambda_val = parameters
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Введите значение параметров.")

    def get_parameters(self):
        k = self.edit_k.text().strip()
        lambda_val = self.edit_c.text().strip()
        if not k or not lambda_val:
            return None
        try:
            k = float(k)
            lambda_val = float(lambda_val)
            if k>=0 and lambda_val >=0:
                return k, lambda_val
            else:
                reply = QMessageBox.question(None, 'Внимание', 'Значение не может быть отрицательным', QMessageBox.Yes, QMessageBox.Yes)
        except ValueError:
            return None

class SetParamentDialog(QDialog):
    def __init__(self, parent=None):
        super(SetParamentDialog, self).__init__(parent)
        self.setWindowTitle("Введите параметры диапазона яркости")

        self.layout = QVBoxLayout(self)
        
        self.label_k = QLabel("Введите стартовое значение:")
        self.edit_k = QLineEdit()
        self.edit_k.setValidator(QDoubleValidator())

        self.label_c = QLabel("Введите конечное значение:")
        self.edit_c = QLineEdit()
        self.edit_c.setValidator(QDoubleValidator())

        self.button_ok = QPushButton("OK")
        self.button_ok.clicked.connect(self.check_and_accept)

        self.layout.addWidget(self.label_k)
        self.layout.addWidget(self.edit_k)
        self.layout.addWidget(self.label_c)
        self.layout.addWidget(self.edit_c)
        self.layout.addWidget(self.button_ok)
            
    def check_and_accept(self):
        parameters = self.get_parameters()
        if parameters is not None:
            start, finish = parameters
            self.accept()
        else:
            QMessageBox.warning(self, "Предупреждение", "Введите значение параметров.")

    def get_parameters(self):
        start = self.edit_k.text().strip()
        finish = self.edit_c.text().strip()
        if not start or not finish:
            return None
        try:
            start = float(start)
            finish = float(finish)
            if start >=0 and finish>=0:
                return start, finish
            else:
                reply = QMessageBox.question(None, 'Внимание', 'Значение не может быть отрицательным', QMessageBox.Yes, QMessageBox.Yes)
        except ValueError:
            return None
    
class ImageWindow(QMainWindow):
    def __init__(self, image_path):
        super(ImageWindow, self).__init__()

        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        self.image = QPixmap(image_path)
        self.label.setPixmap(self.image)

        self.original_image = self.image
        self.noisy_image = self.image

        self.label.setMouseTracking(True)
        self.label.mouseMoveEvent = self.mouseMoveEvent
     
    def reset_to_original(self):
        self.image = QPixmap(self.original_image)
        self.label.setPixmap(self.image)
        
    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image.save(file_path)

    def add_noise(self):
        if hasattr(self, 'image') and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()

            width, height = image_qimage.width(), image_qimage.height()
            noisy_image = QImage(width, height, QImage.Format_ARGB32)
            dialog = WindowDialogAddNoise(self.window())
            result = dialog.exec_()

            if result == QDialog.Accepted:
                noise_factor = dialog.get_parameters()
            else:
                return

            for y in range(height):
                for x in range(width):
                    pixel_color = QColor(image_qimage.pixelColor(x, y))

                    noisy_red = max(0, min(255, int(pixel_color.red() + random.uniform(-noise_factor * 255, noise_factor * 255))))
                    noisy_green = max(0, min(255, int(pixel_color.green() + random.uniform(-noise_factor * 255, noise_factor * 255))))
                    noisy_blue = max(0, min(255, int(pixel_color.blue() + random.uniform(-noise_factor * 255, noise_factor * 255))))

                    noisy_image.setPixel(x, y, QColor(noisy_red, noisy_green, noisy_blue).rgb())

            self.image = QPixmap(noisy_image)
            self.label.setPixmap(self.image)
            self.noisy_image = self.image

    
    def rectangular_filter(self):
        if hasattr(self, 'image') and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()
            
            dialog = WindowDialogKernel(self.window())
            result = dialog.exec_()

            if result == QDialog.Accepted:
                kernel_size = dialog.get_parameters()
            else:
                return

            kernel = [[1] * kernel_size for _ in range(kernel_size)]

            width, height = image_qimage.width(), image_qimage.height()
            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    neighborhood = self.get_neighborhood_for_rectangular(image_qimage, x, y, kernel)

                    avg_red = sum(pixel[0] for pixel in neighborhood) // len(neighborhood)
                    avg_green = sum(pixel[1] for pixel in neighborhood) // len(neighborhood)
                    avg_blue = sum(pixel[2] for pixel in neighborhood) // len(neighborhood)

                    new_image.setPixel(x, y, QColor(avg_red, avg_green, avg_blue).rgb())

            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def get_neighborhood_for_rectangular(self, image, x, y, kernel):
        neighborhood = []
        kernel_size = len(kernel)
        half_kernel = kernel_size // 2

        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                nx, ny = x + i, y + j

                if 0 <= nx < image.width() and 0 <= ny < image.height():
                    neighborhood.append(image.pixelColor(nx, ny).getRgb()[:3])

        return neighborhood
    
    def get_neighborhood_for_median(self, image, x, y, kernel):
        neighborhood = []
        kernel_size = len(kernel)
        half_kernel = kernel_size // 2

        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                nx, ny = x + i, y + j

                if 0 <= nx < image.width() and 0 <= ny < image.height():
                    neighborhood.append(image.pixelColor(nx, ny).getRgb()[:3])

        return neighborhood
    
    def median_filter(self):
        if hasattr(self, 'image') and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()
            
            dialog = WindowDialogKernel(self.window())
            result = dialog.exec_()

            if result == QDialog.Accepted:
                kernel_size = dialog.get_parameters()
            else:
                return

            kernel = [[1] * kernel_size for _ in range(kernel_size)]

            width, height = image_qimage.width(), image_qimage.height()
            new_image = QImage(width, height, QImage.Format_ARGB32)

            for y in range(height):
                for x in range(width):
                    neighborhood = self.get_neighborhood_for_median(image_qimage, x, y, kernel)

                    median_red = int(np.median([pixel[0] for pixel in neighborhood]))
                    median_green = int(np.median([pixel[1] for pixel in neighborhood]))
                    median_blue = int(np.median([pixel[2] for pixel in neighborhood]))

                    new_image.setPixel(x, y, QColor(median_red, median_green, median_blue).rgb())

            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)
    

    def gaussian_filter(self):
            if hasattr(self, 'image') and isinstance(self.image, QPixmap):
                image_qimage = self.image.toImage()
                dialog = WindowDialogSigmaForGauss(self.window())
                result = dialog.exec_()

                if result == QDialog.Accepted:
                    sigma = dialog.get_parameters()
                else:
                    return
                size = int(6 * sigma + 1)
                if size % 2 == 0:
                    size += 1

                kernel = np.zeros((size, size))

                for x in range(size):
                    for y in range(size):
                        kernel[x, y] = self.gaussian_function(x - size // 2, y - size // 2, sigma)

                kernel /= kernel.sum()

                width, height = image_qimage.width(), image_qimage.height()
                new_image = QImage(width, height, QImage.Format_ARGB32)

                for y in range(height):
                    for x in range(width):
                        neighborhood = self.get_neighborhood_for_gaussian(image_qimage, x, y, kernel)
                        new_pixel_value = sum(np.array(pixel) * kernel_value for pixel, kernel_value in zip(neighborhood, kernel.flatten()))

                        new_image.setPixel(x, y, QColor(*new_pixel_value.astype(int)).rgb())

                self.image = QPixmap(new_image)
                self.label.setPixmap(self.image)

    def gaussian_function(self, x, y, sigma):
        return (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))

    def get_neighborhood_for_gaussian(self, image, x, y, kernel):
        neighborhood = []
        kernel_size = len(kernel)
        half_kernel = kernel_size // 2

        for i in range(-half_kernel, half_kernel + 1):
            for j in range(-half_kernel, half_kernel + 1):
                nx, ny = x + i, y + j

                if 0 <= nx < image.width() and 0 <= ny < image.height():
                    neighborhood.append(image.pixelColor(nx, ny).getRgb()[:3])

        return neighborhood

    def sigma_filter(self):
        if hasattr(self, 'image') and isinstance(self.image, QPixmap):
            image_qimage = self.image.toImage()
            dialog = WindowDialogSigma(self.window())
            result = dialog.exec_()

            if result == QDialog.Accepted:
                sigma, k = dialog.get_parameters()
            else:
                return
            
            width, height = image_qimage.width(), image_qimage.height()
            new_image = QImage(width, height, QImage.Format_ARGB32)

            pixels = np.array([[image_qimage.pixelColor(x, y).getRgb()[:3]
                               for x in range(width)] for y in range(height)], dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                neighborhood = self.get_neighborhood(pixels, x, y, k)
                new_pixel_value = self.apply_sigma_filter(neighborhood, sigma)
                color = QColor(*new_pixel_value)
                new_image.setPixel(x, y, color.rgb())

            self.image = QPixmap(new_image)
            self.label.setPixmap(self.image)

    def get_neighborhood(self, pixels, x, y, k):
        half_size = k
        neighborhood = pixels[max(y-half_size, 0):min(y+half_size+1, pixels.shape[0]),
                              max(x-half_size, 0):min(x+half_size+1, pixels.shape[1])]
        return neighborhood.reshape(-1, 3)

    def apply_sigma_filter(self, neighborhood, sigma):
        neighborhood = np.array(neighborhood)
        
        mean = np.mean(neighborhood, axis=0)
        std = np.std(neighborhood, axis=0)
        threshold_min = mean - std * sigma
        threshold_max = mean + std * sigma

        valid_mask = np.any((neighborhood >= threshold_min) & (neighborhood <= threshold_max), axis=-1)

        valid_pixels = neighborhood[valid_mask]

        if valid_pixels.size > 0:
            sigma_mean = np.mean(valid_pixels, axis=0)
        else:
            sigma_mean = mean

        return tuple(sigma_mean.astype(int))
    
    def visual_evaluation_of_processing_quality(self):
        noisy_image_qimage = self.noisy_image.toImage()
        processed_image_qimage = self.image.toImage()
        
        width, height = noisy_image_qimage.width(), noisy_image_qimage.height()
        difference_image = QImage(width, height, QImage.Format_ARGB32)
        
        for y in range(height):
            for x in range(width):
                noisy_pixel = noisy_image_qimage.pixelColor(x, y).getRgb()
                processed_pixel = processed_image_qimage.pixelColor(x, y).getRgb()
                
                diff_red = abs(noisy_pixel[0] - processed_pixel[0])
                diff_green = abs(noisy_pixel[1] - processed_pixel[1])
                diff_blue = abs(noisy_pixel[2] - processed_pixel[2])
                
                difference_image.setPixel(x, y, QColor(diff_red, diff_green, diff_blue).rgb())
        
        self.image = QPixmap.fromImage(difference_image)
        self.label.setPixmap(self.image)

    def unsharp_masking(self, radius=1, threshold=0):
        radius = 1
        threshold = 0

        dialog = ParametersDialog(self.window())
        result = dialog.exec_()

        if result == QDialog.Accepted:
            k, lambda_val = dialog.get_parameters()

            blur_effect = QGraphicsBlurEffect()
            blur_effect.setBlurRadius(radius)
            self.label.setGraphicsEffect(blur_effect)

            img = self.original_image.toImage() 
            blurred_img = self.label.pixmap().toImage()

            sharpness_before = calculate_sharpness(img)

            for x in range(img.width()):
                for y in range(img.height()):
                    pixel_color = QColor(img.pixel(x, y))
                    blurred_pixel_color = QColor(blurred_img.pixel(x, y))
                    
                    new_intensity = pixel_color.lightness() + lambda_val * (pixel_color.lightness() - blurred_pixel_color.lightness()) * k
                    new_intensity = int(min(max(0, new_intensity), 255))
                    
                    if abs(pixel_color.lightness() - blurred_pixel_color.lightness()) >= threshold:
                        new_color = QColor.fromHsl(pixel_color.hue(), pixel_color.saturation(), new_intensity)
                        img.setPixelColor(x, y, new_color)

            sharpness_after = calculate_sharpness(img)
            print("Применяем критерий градиента яркости пикселей")
            if sharpness_after > sharpness_before:
                print('Изображение стало более резким')
            else:
                print('Изображение не стало более резким')

            self.image = QPixmap.fromImage(img)
            self.label.setPixmap(self.image)
         
    def clip_range_transform_image(self, min_intensity=100, max_intensity=200, constant_value=0, keep_original=False):

        dialog = SetParamentDialog(self.window())
        result = dialog.exec_()

        if result == QDialog.Accepted:
            min_intensity, max_intensity = dialog.get_parameters()
            reply = QMessageBox.question(None, 'Подтверждение', 'Хотите ли привести пиксели вне диапазона к произвольному константному значению? Если же нет, то пиксели будут сохранены в исходном виде', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:

                keep_original=False
                dialog_second = WindowDialog(self.window())
                result_second = dialog_second.exec_()

                if result_second == QDialog.Accepted:
                    constant_value = dialog_second.get_parameters()
                    print(constant_value)
            else:
                keep_original= True

            if self.image is not None:
                img = self.image.toImage()
                for x in range(img.width()):
                    for y in range(img.height()):
                        pixel_color = QColor(img.pixel(x, y))
                        intensity = pixel_color.lightness()
                        if intensity < min_intensity or intensity > max_intensity:
                            if keep_original:
                                continue
                            else:
                                new_color = QColor.fromHsl(pixel_color.hue(), pixel_color.saturation(), int(constant_value))
                        else:
                            new_color = pixel_color
                        img.setPixelColor(x, y, new_color)

                self.image = QPixmap.fromImage(img)
                self.label.setPixmap(self.image)
            else:
                QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")

    def binary_transform_image(self, threshold):
        dialog = WindowDialog(self.window())
        result = dialog.exec_()

        if result == QDialog.Accepted:
            threshold = dialog.get_parameters()

            if self.image is not None:
                img = self.image.toImage()
                for x in range(img.width()):
                    for y in range(img.height()):
                        pixel_color = QColor(img.pixel(x, y))
                        intensity = pixel_color.lightness()
                        new_intensity = 255 if intensity > threshold else 0
                        new_color = QColor.fromHsl(pixel_color.hue(), pixel_color.saturation(), new_intensity)
                        img.setPixelColor(x, y, new_color)

                self.image = QPixmap.fromImage(img)
                self.label.setPixmap(self.image)
            else:
                QMessageBox.warning(self, "Предупреждение", "Сначала загрузите изображение")

    def power_transform_image(self):
        dialog = WindowDialog(self.window())
        result = dialog.exec_()

        if result == QDialog.Accepted:
            gamma = dialog.get_parameters()
        
            c = 255 / (self.original_image.width() ** gamma)
            img = self.original_image.toImage()
            for x in range(img.width()):
                for y in range(img.height()):
                    pixel_color = QColor(img.pixel(x, y))
                    intensity = pixel_color.lightness()
                    new_intensity = c * intensity ** gamma
                    new_intensity = int(min(max(0, new_intensity), 255))
                    new_color = QColor.fromHsl(pixel_color.hue(), pixel_color.saturation(), new_intensity)
                    img.setPixelColor(x, y, new_color)

            self.image = QPixmap.fromImage(img)
            self.label.setPixmap(self.image)

    def log_transform_image(self):
        c = 255 / (math.log(1 + self.original_image.width()))
        img = self.original_image.toImage()
        for x in range(img.width()):
            for y in range(img.height()):
                pixel_color = QColor(img.pixel(x, y))
                intensity = pixel_color.lightness()
                new_intensity = c * math.log(1 + intensity)
                new_intensity = int(min(max(0, new_intensity), 255))
                new_color = QColor.fromHsl(pixel_color.hue(), pixel_color.saturation(), new_intensity)
                img.setPixelColor(x, y, new_color)

        self.image = QPixmap.fromImage(img)
        self.label.setPixmap(self.image)

    def mouseMoveEvent(self, event):
        if not self.image.rect().contains(event.pos()):
            return

        if not hasattr(self, 'info_window'):
            self.info_window = QWidget()
            self.info_window.setWindowTitle("Pixel Info")
            layout = QVBoxLayout()

            self.info_label = QLabel()
            layout.addWidget(self.info_label)

            button_layouts = [QHBoxLayout() for _ in range(4)]

            buttons_data1 = [
                ("Сохранить изображение", self.save_image),
                ("Сброс", self.reset_to_original),
            ]

            buttons_data2 = [
                ("Логарифмическое преобразование", self.log_transform_image),
                ("Степенное преобразование", self.power_transform_image),
                ("Бинарное преобразование",  self.binary_transform_image),
                ("Вырезание диапазона яркостей", self.clip_range_transform_image),
            ]

            buttons_data3 = [
                ("Прямоугольный фильтр", self.rectangular_filter),
                ("Медианный фильтр", self.median_filter),
                ("Фильтр гаусса", self.gaussian_filter),
                ("Сигма фильтр", self.sigma_filter),
                ("Нахождение разницы", self.visual_evaluation_of_processing_quality),
                ("Добавление шума", self.add_noise),
            ]

            buttons_data4 = [
                ("Нерезкое маскирование", self.unsharp_masking),
            ]

            for i, (button_text, button_function) in enumerate(buttons_data1):
                button = QPushButton(button_text)
                button.setFixedSize(250, 50)
                button.clicked.connect(button_function)
                button_layouts[0].addWidget(button)

            for i, (button_text, button_function) in enumerate(buttons_data2):
                button = QPushButton(button_text)
                button.setFixedSize(250, 50)
                button.clicked.connect(button_function)
                button_layouts[1].addWidget(button)

            for i, (button_text, button_function) in enumerate(buttons_data3):
                button = QPushButton(button_text)
                button.setFixedSize(250, 50)
                button.clicked.connect(button_function)
                button_layouts[2].addWidget(button)

            for i, (button_text, button_function) in enumerate(buttons_data4):
                button = QPushButton(button_text)
                button.setFixedSize(250, 50)
                button.clicked.connect(button_function)
                button_layouts[3].addWidget(button)

            layout.addWidget(QLabel("Цветность", alignment=Qt.AlignCenter))
            layout.addLayout(button_layouts[1])
            layout.addWidget(QLabel("Сглаживание", alignment=Qt.AlignCenter))
            layout.addLayout(button_layouts[2])
            layout.addWidget(QLabel("Резкость", alignment=Qt.AlignCenter))
            layout.addLayout(button_layouts[3])
            layout.addWidget(QLabel("Управление картинкой", alignment=Qt.AlignCenter))
            layout.addLayout(button_layouts[0])

            self.info_window.setLayout(layout)
            self.info_window.setGeometry(100, 100, 1500, 150)
            self.info_window.show()
                
app = QApplication(sys.argv)

locale = QLocale(QLocale.English)
locale.setNumberOptions(QLocale.RejectGroupSeparator | QLocale.OmitGroupSeparator)
QLocale.setDefault(locale)

file_dialog = QFileDialog()
file_dialog.setFileMode(QFileDialog.ExistingFile)
file_dialog.setNameFilter("Images (*.png *.jpg *.bmp)")
file_path, _ = file_dialog.getOpenFileName()

if file_path:
    window = ImageWindow(file_path)
    window.show()

sys.exit(app.exec_())