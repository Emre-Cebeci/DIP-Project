from PyQt5.QtWidgets import (QApplication, QMainWindow, QMdiArea, 
                             QMdiSubWindow, QAction, QMenu, QLabel, 
                             QVBoxLayout, QWidget, QDialog, QLineEdit, 
                             QPushButton, QHBoxLayout, QFileDialog, 
                             QMessageBox, QScrollArea, QGraphicsPixmapItem, 
                             QGraphicsView, QGraphicsScene, QSplitter, 
                             QFormLayout, QCheckBox, QComboBox, QSpinBox,
                             QGridLayout, QToolBar)

from PyQt5.QtGui import (QImage, QPixmap, QIntValidator, QPainter, QColor, 
                         qBlue, qGreen, qRed, QBrush, QValidator, QIcon)

from PyQt5.QtCore import Qt, QPoint, QSize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import cv2

from simage import SImage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.original_image: SImage = None
        self.current_image: SImage = None

        self.setWindowTitle("Dijital Görüntü İşleme - 221229015 - Emre Cebeci")
        self.setGeometry(100, 100, 800, 600)


        self.create_menu_actions()
        self.create_menus()

        self.create_toolbar_actions()
        self.create_toolbar()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        
        self.splitter = QSplitter(Qt.Horizontal)

        self.left_widget = QWidget()
        self.right_widget = QScrollArea()

        self.left_layout = QVBoxLayout(self.left_widget)
        self.right_layout = QVBoxLayout(self.right_widget)


        self.image_view = DraggableImageView()
        self.image_view.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.image_view)

        self.details_widget = ImageDetailsWidget()
        self.right_layout.addWidget(self.details_widget, 0)
        self.right_layout.addStretch(1)


        self.splitter.addWidget(self.left_widget)
        self.splitter.addWidget(self.right_widget)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([600, 200])

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.splitter)

    def create_menu_actions(self):
        self.new_action = QAction("Yeni", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.triggered.connect(self.create_new_image)

        self.open_action = QAction("Aç", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.open_image)

        self.apply_grayscale_action = QAction("Gri Tonlama")
        self.apply_grayscale_action.setDisabled(True)
        self.apply_grayscale_action.triggered.connect(self.apply_grayscale)

        self.apply_color_space_change_action = QAction("Renk Uzayı Değiştir")
        self.apply_color_space_change_action.setDisabled(True)
        self.apply_color_space_change_action.triggered.connect(self.apply_color_space_change)

        self.apply_gaussian_blur_action = QAction("Gaussian Blur Uygula")
        self.apply_gaussian_blur_action.setDisabled(True)
        self.apply_gaussian_blur_action.triggered.connect(self.apply_gaussian_blur)

        self.apply_histogram_equalizer_action = QAction("Histogram Eşitleme")
        self.apply_histogram_equalizer_action.setDisabled(True)
        self.apply_histogram_equalizer_action.triggered.connect(self.apply_histogram_equalizer)

        self.apply_binary_thresholding_action = QAction("Eşikleme")
        self.apply_binary_thresholding_action.setDisabled(True)
        self.apply_binary_thresholding_action.triggered.connect(self.apply_binary_thresholding)

        self.show_controls_action = QAction("Kontroller")
        self.show_controls_action.triggered.connect(self.show_controls)

    def create_toolbar_actions(self):
        self.undo_action = QAction(QIcon("./assets/icons/undo.png"), "Geri Al", self)
        self.undo_action.setDisabled(True)
        self.undo_action.triggered.connect(self.undo)

    def create_toolbar(self):
        self.toolbar = QToolBar("Main Toolbar")
        self.addToolBar(self.toolbar)
        self.toolbar.setIconSize(QSize(16, 16))

        self.toolbar.addAction(self.undo_action)

    def create_menus(self):
        self.file_menu = self.menuBar().addMenu("Dosya")
        self.file_menu.addAction(self.new_action)
        self.file_menu.addAction(self.open_action)

        self.basic_operations = self.menuBar().addMenu("Ödev 1: Temel İşlevselliği Oluştur")
        self.basic_operations.addAction(self.apply_grayscale_action)
        self.basic_operations.addAction(self.apply_color_space_change_action)
        self.basic_operations.addAction(self.apply_gaussian_blur_action)
        self.basic_operations.addAction(self.apply_histogram_equalizer_action)
        self.basic_operations.addAction(self.apply_binary_thresholding_action)

        self.help = self.menuBar().addMenu("Yardım")
        self.help.addAction(self.show_controls_action)

    def undo(self):
        if self.original_image is None:
            return
        
        self.load_image(self.original_image)

        
# ------------ Start of File Operations ------------

    def create_new_image(self):
        dialog = _SizeInputDialog(self)

        if dialog.exec():
            width, height = dialog.get_width_height()
            img_arr = np.full((height, width, 3), 255, dtype=np.uint8)
            img = SImage(img_arr, "rgb")
            self.load_image(img)

    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Görüntü Aç", "", "Görsel (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        try:
            img = SImage.from_file_path(file_path, "rgb")
            self.load_image(img)
        
        except FileNotFoundError as e:
            self.show_error_dialog(e)

# ------------ End of File Operations ------------


# ------------ Start of Help ------------

    def show_controls(self):
        dialog = _ControlsDialog(self)
        dialog.exec()

# ------------ End of Help ------------


    def load_image(self, image: SImage):
        if image is None or image.image_array.size == 0:
            self.show_error_dialog("Görüntü yüklenirken bir hata oluştu.")
            return
        
        if image.color_space == "gray":
            height, width = image.image_array.shape
            bytes_per_line = width
            q_image = QImage(image.image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        elif image.color_space == "rgb":
            height, width, channels = image.image_array.shape
            bytes_per_line = channels * width
            q_image = QImage(image.image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        elif image.color_space == "hsv":
            height, width, channels = image.image_array.shape
            bytes_per_line = channels * width
            q_image = QImage(image.change_color_space("rgb").image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)

        self.pixmap = QPixmap.fromImage(q_image)
        self.image_view.pixmap_item.setPixmap(self.pixmap)
        self.image_view.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(self.image_view)

        self.original_image = self.current_image
        self.current_image = image
        self.details_widget.image = image

        self.details_widget.update_histogram()
        self.details_widget.update_details()

        self.apply_grayscale_action.setDisabled(False)
        self.apply_color_space_change_action.setDisabled(False)
        self.apply_gaussian_blur_action.setDisabled(False)
        self.apply_histogram_equalizer_action.setDisabled(False)
        self.apply_binary_thresholding_action.setDisabled(False)

        self.undo_action.setDisabled(False)
        
    def show_error_dialog(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Hata")
        error_dialog.setText(message)
        error_dialog.exec_()


# ------------ Start of Basic Operations ------------

    def apply_grayscale(self):
        self.original_image = self.current_image
        img = self.current_image.change_color_space("gray")
        self.load_image(img)

    def apply_color_space_change(self):
        dialog = _ColorSpaceOptionDialog(self)

        if dialog.exec():
            option = dialog.get_selected_option()
            if option is None:
                return
        
        self.original_image = self.current_image
        img = self.current_image.change_color_space(option)
        self.load_image(img)
    
    def apply_gaussian_blur(self):
        dialog = _GaussianBlurKernelDialog(self.current_image, self)

        if dialog.exec():
            kernel_size = dialog.get_selected_kernel_size()
            if kernel_size is None:
                return

        self.original_image = self.current_image
        img = self.current_image.apply_gaussian_blur((kernel_size, kernel_size))
        self.load_image(img)
    
    def apply_histogram_equalizer(self):
        if self.current_image.color_space != "gray":
            self.show_error_dialog("Görüntü gri renk uzayında olmalı.")
            return
        
        self.original_image = self.current_image
        img = self.current_image.apply_histogram_equalizer()
        self.load_image(img)

    def apply_binary_thresholding(self):
        dialog = _ThresholdingDialog(self)

        if dialog.exec():
            threshold_value, max_value = dialog.get_values()
            if threshold_value is None or max_value is None:
                return
        
        self.original_image = self.current_image
        img = self.current_image.apply_binary_thresholding(threshold_value, max_value)
        self.load_image(img)


# ------------ End of Basic Operations ------------

class _ThresholdingDialog(QDialog):
    def __init__(self, image: SImage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Eşikleme İçin Değerler Seç")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.threshold_label = QLabel("Eşik Değeri:")
        self.threshold_value = QSpinBox()

        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(125)
        
        self.max_value_label = QLabel("Maksimum Değer:")
        self.max_value_value = QSpinBox()

        self.max_value_value.setRange(0, 255)
        self.max_value_value.setValue(255)

        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)

        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_value)
        layout.addWidget(self.max_value_label)
        layout.addWidget(self.max_value_value)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_values(self):
        return self.threshold_value.value(), self.max_value_value.value()
    

class _SizeInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Görüntü Boyutlarını Girin")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.width_label = QLabel("Genişlik:")
        self.width_input = QLineEdit()
        self.width_input.setValidator(QIntValidator(1, 10000))

        self.height_label = QLabel("Yükseklik:")
        self.height_input = QLineEdit()
        self.height_input.setValidator(QIntValidator(1, 10000))

        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)

        self.ok_button.setEnabled(False)

        width_layout = QHBoxLayout()
        width_layout.addWidget(self.width_label)
        width_layout.addWidget(self.width_input)

        height_layout = QHBoxLayout()
        height_layout.addWidget(self.height_label)
        height_layout.addWidget(self.height_input)

        layout.addLayout(width_layout)
        layout.addLayout(height_layout)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

        self.width_input.textChanged.connect(self.validate_inputs)
        self.height_input.textChanged.connect(self.validate_inputs)

    def validate_inputs(self):
        width_valid = self.width_input.validator().validate(self.width_input.text(), 0)[0] == QIntValidator.Acceptable
        height_valid = self.height_input.validator().validate(self.height_input.text(), 0)[0] == QIntValidator.Acceptable

        if width_valid and height_valid:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)

    def get_width_height(self):
        return int(self.width_input.text()), int(self.height_input.text())


class _ColorSpaceOptionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Renk Uzayı Seç")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.label = QLabel("Renk Uzayı:")
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Gray", "RGB", "HSV"])

        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)

        layout.addWidget(self.label)
        layout.addWidget(self.combo_box)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_selected_option(self):
        return self.combo_box.currentText().lower()


class _GaussianBlurKernelDialog(QDialog):
    def __init__(self, image: SImage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gaussian Bulanıklık İçin Kernel Boyu Seç")
        self.resize(300, 150)

        layout = QVBoxLayout()

        self.label = QLabel("Kernel Boyu:")
        self.kernel_size = QSpinBox()

        height = image.image_array.shape[0]
        width = image.image_array.shape[1]
        max_size = height if width > height else width

        self.kernel_size.setRange(1, max_size)
        self.kernel_size.setSingleStep(2)
        self.kernel_size.setValue(1)
        
        self.ok_button = QPushButton("Tamam")
        self.ok_button.clicked.connect(self.accept)

        layout.addWidget(self.label)
        layout.addWidget(self.kernel_size)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_selected_kernel_size(self):
        val = self.kernel_size.value()
        
        if val <= 0:
            self.parent.show_error_dialog(self.parent, "Kernel için uygun olmayan bir boyut girildi.")
            return None

        if val % 2 == 0:
            val -= 1

        return val


class _ControlsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kontroller")
        self.resize(300, 150)

        layout = QGridLayout()

        label0 = QLabel("Yakınlaştırma/Uzaklaştırma:")
        ctrl0 = QLabel("Ctrl + Mouse Wheel")

        label1 = QLabel("Dikey kaydırma:")
        ctrl1 = QLabel("Mouse Wheel")

        label2 = QLabel("Yatay kaydırma:")
        ctrl2 = QLabel("Shift + Mouse Wheel")

        label3 = QLabel("Görüntü Sürükleme:")
        ctrl3 = QLabel("Left Mouse Button")

        layout.addWidget(label0, 0, 0)
        layout.addWidget(ctrl0, 0, 1)
        layout.addWidget(label1, 1, 0)
        layout.addWidget(ctrl1, 1, 1)
        layout.addWidget(label2, 2, 0)
        layout.addWidget(ctrl2, 2, 1)
        layout.addWidget(label3, 3, 0)
        layout.addWidget(ctrl3, 3, 1)

        self.setLayout(layout)


class DraggableImageView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.scale_factor = 1.0

        self.pixmap = QPixmap()
        self.pixmap_item.setPixmap(self.pixmap)

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)

        self.dragging = False
        self.drag_start_pos = QPoint()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ShiftModifier:
            angle = event.angleDelta().y()
            self.horizontal_scroll(angle)

        elif event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            self.zoom_image(angle)

        else:
            super().wheelEvent(event)

    def zoom_image(self, angle):
        if angle > 0:
            self.scale_factor *= 1.1

        else:
            self.scale_factor /= 1.1

        self.pixmap_item.setScale(self.scale_factor)

    def horizontal_scroll(self, delta):
        current_pos = self.pixmap_item.pos()
        new_pos = QPoint(int(current_pos.x() + delta), int(current_pos.y()))
        self.pixmap_item.setPos(new_pos)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.drag_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.pos() - self.drag_start_pos
            self.pixmap_item.moveBy(delta.x(), delta.y())
            self.drag_start_pos = event.pos()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)


class ImageDetailsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.image: SImage = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.details_layout = QFormLayout()
        self.label_size = QLabel("-")
        self.label_depth = QLabel("-")

        self.details_layout.addRow("Boyut:", self.label_size)
        self.details_layout.addRow("Renk Derinliği:", self.label_depth)

        self.layout.addLayout(self.details_layout)

        self.hist_rgb_checkbox_layout = QHBoxLayout()
        self.hist_rgb_cb_gray = QCheckBox("Grayscale")
        self.hist_rgb_cb_red = QCheckBox("Red")
        self.hist_rgb_cb_green = QCheckBox("Green")
        self.hist_rgb_cb_blue = QCheckBox("Blue")

        self.hist_rgb_cb_gray.setChecked(True)
        self.hist_rgb_cb_red.setChecked(True)
        self.hist_rgb_cb_green.setChecked(True)
        self.hist_rgb_cb_blue.setChecked(True)

        self.hist_rgb_cb_gray.stateChanged.connect(self.update_histogram)
        self.hist_rgb_cb_red.stateChanged.connect(self.update_histogram)
        self.hist_rgb_cb_green.stateChanged.connect(self.update_histogram)
        self.hist_rgb_cb_blue.stateChanged.connect(self.update_histogram)

        self.hist_rgb_checkbox_layout.addWidget(self.hist_rgb_cb_gray)
        self.hist_rgb_checkbox_layout.addWidget(self.hist_rgb_cb_red)
        self.hist_rgb_checkbox_layout.addWidget(self.hist_rgb_cb_green)
        self.hist_rgb_checkbox_layout.addWidget(self.hist_rgb_cb_blue)

        self.layout.addLayout(self.hist_rgb_checkbox_layout)
        self._hide_layout(self.hist_rgb_checkbox_layout)


        self.hist_hsv_checkbox_layout = QHBoxLayout()
        self.hist_hsv_cb_gray = QCheckBox("Grayscale")
        self.hist_hsv_cb_hue = QCheckBox("Hue")
        self.hist_hsv_cb_saturation = QCheckBox("Saturation")
        self.hist_hsv_cb_value = QCheckBox("Value")

        self.hist_hsv_cb_gray.setChecked(True)
        self.hist_hsv_cb_hue.setChecked(True)
        self.hist_hsv_cb_saturation.setChecked(True)
        self.hist_hsv_cb_value.setChecked(True)

        self.hist_hsv_cb_gray.stateChanged.connect(self.update_histogram)
        self.hist_hsv_cb_hue.stateChanged.connect(self.update_histogram)
        self.hist_hsv_cb_saturation.stateChanged.connect(self.update_histogram)
        self.hist_hsv_cb_value.stateChanged.connect(self.update_histogram)

        self.hist_hsv_checkbox_layout.addWidget(self.hist_hsv_cb_gray)
        self.hist_hsv_checkbox_layout.addWidget(self.hist_hsv_cb_hue)
        self.hist_hsv_checkbox_layout.addWidget(self.hist_hsv_cb_saturation)
        self.hist_hsv_checkbox_layout.addWidget(self.hist_hsv_cb_value)

        self.layout.addLayout(self.hist_hsv_checkbox_layout)
        self._hide_layout(self.hist_hsv_checkbox_layout)


        self.figure = Figure(figsize=(16,3))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
    def update_details(self):
        if self.image is None or self.image.image_array.size == 0:
            return
        
        height, width = self.image.image_array.shape[:2]
        channels = self.image.image_array.shape[2] if len(self.image.image_array.shape) == 3 else 1
        bit_depth = self.image.image_array.dtype.itemsize * 8

        self.label_size.setText(f"{width} x {height}")
        self.label_depth.setText(f"{bit_depth * channels}-bit")

    def update_histogram(self):
        if self.image is None or self.image.image_array.size == 0:
            return

        if self.image.color_space == "rgb":
            self._hide_layout(self.hist_hsv_checkbox_layout)
            self._show_layout(self.hist_rgb_checkbox_layout)

            gray_img = cv2.cvtColor(self.image.image_array, cv2.COLOR_RGB2GRAY)

            hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
            hist_blue = cv2.calcHist([self.image.image_array[:, :, 0]], [0], None, [256], [0, 256]).flatten()
            hist_green = cv2.calcHist([self.image.image_array[:, :, 1]], [0], None, [256], [0, 256]).flatten()
            hist_red = cv2.calcHist([self.image.image_array[:, :, 2]], [0], None, [256], [0, 256]).flatten()

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if self.hist_rgb_cb_gray.isChecked():
                ax.fill_between(range(256), hist_gray, color="black", alpha=0.25)
            if self.hist_rgb_cb_red.isChecked():
                ax.fill_between(range(256), hist_red, color="red", alpha=0.25)
            if self.hist_rgb_cb_green.isChecked():
                ax.fill_between(range(256), hist_green, color="green", alpha=0.25)
            if self.hist_rgb_cb_blue.isChecked():
                ax.fill_between(range(256), hist_blue, color="blue", alpha=0.25)
            
            ax.set_title("Histogram")
            ax.set_xlim([0, 255])
            self.canvas.draw()
            
        elif self.image.color_space == "hsv":
            self._hide_layout(self.hist_rgb_checkbox_layout)
            self._show_layout(self.hist_hsv_checkbox_layout)

            if (self.hist_hsv_checkbox_layout.parentWidget() is None):
                self.layout.addLayout(self.hist_hsv_checkbox_layout)
            
            gray_img = cv2.cvtColor(self.image.image_array, cv2.COLOR_HSV2RGB)
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

            hist_gray = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
            hist_hue = cv2.calcHist([self.image.image_array[:, :, 0]], [0], None, [256], [0, 256]).flatten()
            hist_sat = cv2.calcHist([self.image.image_array[:, :, 1]], [0], None, [256], [0, 256]).flatten()
            hist_val = cv2.calcHist([self.image.image_array[:, :, 2]], [0], None, [256], [0, 256]).flatten()

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            if self.hist_hsv_cb_gray.isChecked():
                ax.fill_between(range(256), hist_gray, color="black", alpha=0.25)
            if self.hist_hsv_cb_hue.isChecked():
                ax.fill_between(range(256), hist_hue, color="purple", alpha=0.25)
            if self.hist_hsv_cb_saturation.isChecked():
                ax.fill_between(range(256), hist_sat, color="yellow", alpha=0.25)
            if self.hist_hsv_cb_value.isChecked():
                ax.fill_between(range(256), hist_val, color="blue", alpha=0.25)

            ax.set_title("Histogram")
            ax.set_xlim([0, 255])
            self.canvas.draw()

        elif self.image.color_space == "gray":
            self._hide_layout(self.hist_rgb_checkbox_layout)
            self._hide_layout(self.hist_hsv_checkbox_layout)

            hist_gray = cv2.calcHist([self.image.image_array], [0], None, [256], [0, 256]).flatten()

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.fill_between(range(256), hist_gray, color="black", alpha=0.25)
            ax.set_title("Histogram")
            ax.set_xlim([0, 255])
            self.canvas.draw()
        
    def _hide_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.hide()

    def _show_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()
            if widget:
                widget.show()
