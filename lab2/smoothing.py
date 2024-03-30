import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QInputDialog,
)
from PyQt5.QtGui import QPixmap, QImage


class ImageSmoothingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Smoothing App")
        self.image_label = QLabel(self)
        self.result_label = QLabel(self)
        self.load_button = QPushButton("Load Image", self)
        self.load_button.clicked.connect(self.load_image)

        self.reset_image_button = QPushButton("Reset Image", self)
        self.reset_image_button.clicked.connect(self.reset_image)

        self.rect_filter_button = QPushButton("Apply Rectangular Filter (3x3)", self)
        self.rect_filter_button.clicked.connect(
            lambda: self.apply_filter(self.apply_rectangular_filter, size=3)
        )

        self.median_filter_button = QPushButton("Apply Median Filter (3x3)", self)
        self.median_filter_button.clicked.connect(
            lambda: self.apply_filter(self.apply_median_filter, size=3)
        )

        self.rect_filter_button_5x5 = QPushButton(
            "Apply Rectangular Filter (5x5)", self
        )
        self.rect_filter_button_5x5.clicked.connect(
            lambda: self.apply_filter(self.apply_rectangular_filter, size=5)
        )

        self.median_filter_button_5x5 = QPushButton("Apply Median Filter (5x5)", self)
        self.median_filter_button_5x5.clicked.connect(
            lambda: self.apply_filter(self.apply_median_filter, size=5)
        )

        self.gaussian_filter_button = QPushButton("Apply Gaussian Filter", self)
        self.gaussian_filter_button.clicked.connect(
            self.apply_gaussian_filter_with_dialog
        )

        self.sigma_filter_button = QPushButton("Apply Sigma Filter", self)
        self.sigma_filter_button.clicked.connect(self.apply_sigma_filter_with_dialog)

        self.calculate_sharpness_button = QPushButton("Calculate Sharpness", self)
        self.calculate_sharpness_button.clicked.connect(self.calculate_sharpness)

        self.add_noise_button = QPushButton("Add Noise", self)
        self.add_noise_button.clicked.connect(self.add_noise)

        self.save_image_button = QPushButton("Save current image", self)
        self.save_image_button.clicked.connect(self.save_image)

        grid = QGridLayout()
        grid.addWidget(self.load_button, 0, 0)
        grid.addWidget(self.reset_image_button, 0, 1)
        grid.addWidget(self.rect_filter_button, 1, 0)
        grid.addWidget(self.median_filter_button, 1, 1)
        grid.addWidget(self.gaussian_filter_button, 2, 0)
        grid.addWidget(self.sigma_filter_button, 2, 1)
        grid.addWidget(self.calculate_sharpness_button, 3, 0)
        grid.addWidget(self.add_noise_button, 3, 1)
        grid.addWidget(self.rect_filter_button_5x5, 4, 0)
        grid.addWidget(self.median_filter_button_5x5, 4, 1)
        grid.addWidget(self.save_image_button, 5, 0)
        grid.addWidget(self.image_label, 6, 0, 1, 2)
        grid.addWidget(self.result_label, 7, 0, 1, 2)
        self.setLayout(grid)
        self.image = None
        self.smoothed_image = None

        self.rect_filter_button.setVisible(False)
        self.median_filter_button.setVisible(False)
        self.gaussian_filter_button.setVisible(False)
        self.sigma_filter_button.setVisible(False)
        self.calculate_sharpness_button.setVisible(False)
        self.add_noise_button.setVisible(False)

        self.rect_filter_button_5x5.setVisible(False)
        self.median_filter_button_5x5.setVisible(False)

    def load_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.raw_image = self.image
            self.display_image(self.image, self.image_label)
            self.rect_filter_button.setVisible(True)
            self.median_filter_button.setVisible(True)
            self.gaussian_filter_button.setVisible(True)
            self.sigma_filter_button.setVisible(True)
            self.calculate_sharpness_button.setVisible(True)
            self.rect_filter_button_5x5.setVisible(True)
            self.median_filter_button_5x5.setVisible(True)
            self.add_noise_button.setVisible(True)

    def display_image(self, image, label_widget):
        height, width = image.shape[:2]
        bytes_per_line = width
        q_img = QImage(
            image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap(q_img)
        pixmap = pixmap.scaledToWidth(400)
        if label_widget is self.image_label:
            self.image = image
        elif label_widget is self.result_label:
            self.smoothed_image = image
        label_widget.setPixmap(pixmap)

    def reset_image(self):
        self.image = self.raw_image
        self.display_image(self.image, self.image_label)

    def add_noise(self):
        noise = np.random.randint(0, int(256 * 0.3), size=self.image.shape, dtype=np.uint8)  # Учитываем диапазон интенсивности пикселей (0-255)
        self.image = np.clip(self.image.astype(np.uint16) + noise.astype(np.uint16), 0, 255).astype(np.uint8)  # Преобразуем в 16-битное целое число, чтобы избежать переполнения
        self.display_image(self.image, self.image_label)

    def apply_filter(self, filter_func, size=None, sigma=None):
        if self.image is None:
            return
        if size:
            self.smoothed_image = filter_func(self.image, size)
        elif sigma:
            self.smoothed_image = filter_func(self.image, sigma)
        self.display_image(self.smoothed_image, self.image_label)

    def apply_rectangular_filter(self, image, size):
        height, width = image.shape
        smoothed_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                roi = image[
                    max(0, i - size // 2) : min(height, i + size // 2 + 1),
                    max(0, j - size // 2) : min(width, j + size // 2 + 1),
                ]
                smoothed_image[i, j] = np.mean(roi)
        return smoothed_image

    def apply_median_filter(self, image, size):
        height, width = image.shape
        smoothed_image = np.zeros((height, width), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                roi = image[
                    max(0, i - size // 2) : min(height, i + size // 2 + 1),
                    max(0, j - size // 2) : min(width, j + size // 2 + 1),
                ]
                smoothed_image[i, j] = np.median(roi)
        return smoothed_image

    def apply_gaussian_filter_with_dialog(self):
        sigma, ok = QInputDialog.getDouble(self, "Sigma", "Enter sigma", value=1)
        if ok:
            self.apply_filter(self.apply_gaussian_filter, sigma=sigma)

    def apply_sigma_filter_with_dialog(self):
        sigma, ok = QInputDialog.getDouble(self, "Sigma", "Enter sigma", value=1)
        if ok:
            self.apply_filter(self.apply_sigma_filter, sigma=sigma)

    def generate_gaussian_kernel(self, size, sigma):
        kernel = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * sigma**2))
            * np.exp(
                -((x - size // 2) ** 2) / (2 * sigma**2)
                - (y - size // 2) ** 2 / (2 * sigma**2)
            ),
            (size, size),
        )
        return kernel / np.sum(kernel)

    def apply_gaussian_filter(self, image, sigma):
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = self.generate_gaussian_kernel(kernel_size, sigma)
        padded_image = np.pad(
            image,
            (
                (kernel_size // 2, kernel_size // 2),
                (kernel_size // 2, kernel_size // 2),
            ),
            mode="constant",
        )

        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(
                    kernel * padded_image[i : i + kernel_size, j : j + kernel_size]
                )

        return result

    def apply_sigma_filter(self, image, sigma):
        blurred = self.apply_gaussian_filter(image, sigma)
        diff_mask = np.abs(image - blurred) < 10 * sigma
        smoothed_image = np.where(diff_mask, blurred, image)

        return smoothed_image

    def calculate_sharpness(self):
        if self.image is None:
            return

        sobelx = self.sobel_filter(self.image, "x")
        sobely = self.sobel_filter(self.image, "y")
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness = np.mean(gradient_magnitude)

        msg = QMessageBox()
        msg.setWindowTitle("Sharpness")
        msg.setText(f"Sharpness: {sharpness}")
        msg.exec_()

    def sobel_filter(self, image, axis):
        if axis == "x":
            kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        elif axis == "y":
            kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        else:
            raise ValueError("Invalid axis parameter")

        # Apply the kernel to the image using convolution
        filtered_image = np.zeros_like(image, dtype=np.float32)
        height, width = image.shape
        k_height, k_width = kernel.shape
        pad_height = k_height // 2
        pad_width = k_width // 2
        padded_image = np.pad(
            image, ((pad_height, pad_height), (pad_width, pad_width)), mode="constant"
        )

        for i in range(height):
            for j in range(width):
                region = padded_image[i : i + k_height, j : j + k_width]
                filtered_value = np.sum(region * kernel)
                filtered_image[i, j] = filtered_value

        return filtered_image

    def save_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            height, width = self.image.shape[:2]
            bytes_per_line = width
            q_img = QImage(
                self.image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
            pixmap = QPixmap(q_img)
            pixmap.save(file_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSmoothingApp()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())
