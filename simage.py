import numpy as np
import cv2
import copy
from PyQt5.QtGui import QImage
from smatrix import SMatrix
import math
from utils import SMath
from collections import defaultdict
from openpyxl import Workbook

class SImage():
    def __init__(self, image_array: np.ndarray):
        self.R_matrix = SMatrix.from_nparray(image_array[:, :, 0])
        self.G_matrix = SMatrix.from_nparray(image_array[:, :, 1])
        self.B_matrix = SMatrix.from_nparray(image_array[:, :, 2])
        self.height, self.width = image_array.shape[:2]


    def __add__(self, other):
        if isinstance(other, SImage):

            if self.width != other.width or self.height != other.height:
                raise ValueError("Images must have the same shape to add.")
            
            result = SImage.new_empty_image(self.width, self.height)

            for i in range(self.height):
                for j in range(self.width):
                    result.R_matrix[i, j] = self.R_matrix[i ,j] + other.R_matrix[i, j]
                    result.G_matrix[i, j] = self.G_matrix[i ,j] + other.G_matrix[i, j]
                    result.B_matrix[i, j] = self.B_matrix[i ,j] + other.B_matrix[i, j] 
            
            return result
        
        else:
            raise TypeError("Unsupported operand type for +: 'SImage' and '{}'".format(type(other).__name__))


    @classmethod
    def new_empty_image(cls, width: int, height: int):
        image_array = np.ones((height, width, 3), dtype=np.uint8) * 255;
        return cls(image_array)


    @classmethod
    def from_file_path(cls, file_path: str):
        img = cv2.imread(file_path)
        
        if img is None:
            raise FileNotFoundError(f"Could not read the file: {file_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_array = np.array(img)
        return cls(image_array)
    

    def as_nparray(self):
        height, width = self.R_matrix.shape
        image_array = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                image_array[i][j][0] = self.R_matrix[i, j]
                image_array[i][j][1] = self.G_matrix[i, j]
                image_array[i][j][2] = self.B_matrix[i, j]

        return image_array


    def as_qimage(self):
        np_arr = self.as_nparray()
        height, width, _ = np_arr.shape
        bytes_per_line = 3 * width
        return QImage(np_arr, width, height, bytes_per_line, QImage.Format_RGB888) # type: ignore


    def calculate_histogram(self):
        r_vals = [0] * 256
        g_vals = [0] * 256
        b_vals = [0] * 256
        gray_vals = [0] * 256
        
        for i in range(self.R_matrix.rows):
            for j in range(self.R_matrix.cols):
                r_vals[self.R_matrix[i, j]] += 1
                g_vals[self.G_matrix[i, j]] += 1
                b_vals[self.B_matrix[i, j]] += 1
                gray_vals[int((self.R_matrix[i, j] * 0.2989 + self.G_matrix[i, j] * 0.5870 + self.B_matrix[i, j] * 0.1140))] += 1
        
        return r_vals, g_vals, b_vals, gray_vals


    def resize(self, width, height, interpolation=["nearest", "bilinear", "bicubic"], in_place=False):
        new_img = SImage.new_empty_image(width, height)

        if interpolation == "nearest":
            for i in range(height):
                for j in range(width):
                    x = min(int(i * self.R_matrix.rows / height), self.height - 1)
                    y = min(int(j * self.R_matrix.cols / width), self.width - 1)
                    new_img.R_matrix[i, j] = self.R_matrix[x, y]
                    new_img.G_matrix[i, j] = self.G_matrix[x, y]
                    new_img.B_matrix[i, j] = self.B_matrix[x, y]

        elif interpolation == "bilinear":
            for i in range(height):
                for j in range(width):
                    x = i * self.R_matrix.rows / height if height > 1 else 0
                    y = j * self.R_matrix.cols / width if width > 1 else 0
                    x1, y1 = int(x), int(y)
                    x2, y2 = min(x1 + 1, self.R_matrix.rows - 1), min(y1 + 1, self.R_matrix.cols - 1)

                    a = x - x1
                    b = y - y1

                    new_img.R_matrix[i, j] = min(int(self.R_matrix[x1, y1] * (1 - a) * (1 - b) +
                                               self.R_matrix[x2, y1] * a * (1 - b) +
                                               self.R_matrix[x1, y2] * (1 - a) * b +
                                               self.R_matrix[x2, y2] * a * b), 255)

                    new_img.G_matrix[i, j] = min(int(self.G_matrix[x1, y1] * (1 - a) * (1 - b) +
                                               self.G_matrix[x2, y1] * a * (1 - b) +
                                               self.G_matrix[x1, y2] * (1 - a) * b +
                                               self.G_matrix[x2, y2] * a * b), 255)

                    new_img.B_matrix[i, j] = min(int(self.B_matrix[x1, y1] * (1 - a) * (1 - b) +
                                               self.B_matrix[x2, y1] * a * (1 - b) +
                                               self.B_matrix[x1, y2] * (1 - a) * b +
                                               self.B_matrix[x2, y2] * a * b), 255)
        elif interpolation == "bicubic":
            for i in range(height):
                for j in range(width):
                    src_x = i * (self.R_matrix.rows - 1) / (height - 1) if height > 1 else 0
                    src_y = j * (self.R_matrix.cols - 1) / (width - 1) if width > 1 else 0

                    x0 = int(src_x)
                    y0 = int(src_y)
                    x1 = min(x0 + 1, self.R_matrix.rows - 1)
                    y1 = min(y0 + 1, self.R_matrix.cols - 1)

                    dx = src_x - x0
                    dy = src_y - y0

                    r00 = self.R_matrix[x0, y0]
                    r01 = self.R_matrix[x0, y1]
                    r10 = self.R_matrix[x1, y0]
                    r11 = self.R_matrix[x1, y1]
                    new_img.R_matrix[i, j] = int(
                        r00 * (1 - dx) * (1 - dy) +
                        r10 * dx * (1 - dy) +
                        r01 * (1 - dx) * dy +
                        r11 * dx * dy
                    )

                    g00 = self.G_matrix[x0, y0]
                    g01 = self.G_matrix[x0, y1]
                    g10 = self.G_matrix[x1, y0]
                    g11 = self.G_matrix[x1, y1]
                    new_img.G_matrix[i, j] = int(
                        g00 * (1 - dx) * (1 - dy) +
                        g10 * dx * (1 - dy) +
                        g01 * (1 - dx) * dy +
                        g11 * dx * dy
                    )

                    b00 = self.B_matrix[x0, y0]
                    b01 = self.B_matrix[x0, y1]
                    b10 = self.B_matrix[x1, y0]
                    b11 = self.B_matrix[x1, y1]
                    new_img.B_matrix[i, j] = int(
                        b00 * (1 - dx) * (1 - dy) +
                        b10 * dx * (1 - dy) +
                        b01 * (1 - dx) * dy +
                        b11 * dx * dy
                    )
        else:
            raise ValueError("Invalid interpolation method. Choose from 'nearest', 'bilinear', or 'bicubic'.")

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix
            self.height = new_img.height
            self.width = new_img.width

        return new_img


    def zoom(self, zoom_factor: float, interpolation=["nearest", "bilinear", "bicubic"], in_place=False):
        if zoom_factor <= 0 or zoom_factor > 2:
            raise ValueError("Invalid zoom factor. Zoom factor must be in range (0, 2].")

        if zoom_factor < 1:
            new_height = self.height
            new_width = self.width

            small_h = max(1, int(self.height * zoom_factor))
            small_w = max(1, int(self.width * zoom_factor))

            small_img = self.resize(small_w, small_h, interpolation)

            new_img = SImage.new_empty_image(new_width, new_height)

            start_i = (new_height - small_h) // 2
            start_j = (new_width - small_w) // 2

            for i in range(small_h):
                for j in range(small_w):
                    new_img.R_matrix[start_i + i, start_j + j] = small_img.R_matrix[i, j]
                    new_img.G_matrix[start_i + i, start_j + j] = small_img.G_matrix[i, j]
                    new_img.B_matrix[start_i + i, start_j + j] = small_img.B_matrix[i, j]

        else:
            scaled_width = int(self.width * zoom_factor)
            scaled_height = int(self.height * zoom_factor)
            scaled_image = self.resize(scaled_width, scaled_height, interpolation)

            x0 = (scaled_width - self.width) // 2
            y0 = (scaled_height - self.height) // 2

            new_img = SImage.new_empty_image(self.width, self.height)
            for i in range(self.height):
                for j in range(self.width):
                    new_img.R_matrix[i, j] = scaled_image.R_matrix[y0 + i, x0 + j]
                    new_img.G_matrix[i, j] = scaled_image.G_matrix[y0 + i, x0 + j]
                    new_img.B_matrix[i, j] = scaled_image.B_matrix[y0 + i, x0 + j]

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix
            self.height = new_img.height
            self.width = new_img.width

        return new_img


    def rotate(self, angle, is_in_degrees=False, interpolation=["nearest", "bilinear", "bicubic"], in_place=False):
        if is_in_degrees:
            angle = angle / 180 * math.pi

        cos_theta = math.cos(angle)
        sin_theta = math.sin(angle)

        cx = self.width / 2
        cy = self.height / 2

        new_img = SImage.new_empty_image(self.width, self.height)

        for i in range(self.height):
            for j in range(self.width):
                x = j - cx
                y = i - cy

                x_rot = x * cos_theta - y * sin_theta
                y_rot = x * sin_theta + y * cos_theta

                src_x = x_rot + cx
                src_y = y_rot + cy

                if interpolation == "nearest":
                    sx = int(round(src_y))
                    sy = int(round(src_x))
                    if 0 <= sx < self.height and 0 <= sy < self.width:
                        new_img.R_matrix[i, j] = self.R_matrix[sx, sy]
                        new_img.G_matrix[i, j] = self.G_matrix[sx, sy]
                        new_img.B_matrix[i, j] = self.B_matrix[sx, sy]

                elif interpolation == "bilinear":
                    x1 = int(src_y)
                    y1 = int(src_x)
                    x2 = min(x1 + 1, self.height - 1)
                    y2 = min(y1 + 1, self.width - 1)

                    a = src_y - x1
                    b = src_x - y1

                    if 0 <= x1 < self.height and 0 <= y1 < self.width:
                        def interp(channel, x1, x2, y1, y2, a, b):
                            return (
                                channel[x1, y1] * (1 - a) * (1 - b) +
                                channel[x2, y1] * a * (1 - b) +
                                channel[x1, y2] * (1 - a) * b +
                                channel[x2, y2] * a * b
                            )

                        new_img.R_matrix[i, j] = min(int(interp(self.R_matrix, x1, x2, y1, y2, a, b)), 255)
                        new_img.G_matrix[i, j] = min(int(interp(self.G_matrix, x1, x2, y1, y2, a, b)), 255)
                        new_img.B_matrix[i, j] = min(int(interp(self.B_matrix, x1, x2, y1, y2, a, b)), 255)

                elif interpolation == "bicubic":
                    x = src_y
                    y = src_x
                    ix = int(x)
                    iy = int(y)

                    if 1 <= ix < self.height - 2 and 1 <= iy < self.width - 2:
                        def cubic(t):
                            t = abs(t)
                            if t <= 1:
                                return 1 - 2 * t * t + t * t * t
                            elif t < 2:
                                return 4 - 8 * t + 5 * t * t - t * t * t
                            return 0

                        def bicubic(channel, x, y):
                            value = 0
                            for m in range(-1, 3):
                                for n in range(-1, 3):
                                    xm = ix + m
                                    yn = iy + n
                                    if 0 <= xm < self.height and 0 <= yn < self.width:
                                        weight = cubic(m - (x - ix)) * cubic(n - (y - iy))
                                        value += channel[xm, yn] * weight
                            return max(0, min(int(value), 255))

                        new_img.R_matrix[i, j] = bicubic(self.R_matrix, x, y)
                        new_img.G_matrix[i, j] = bicubic(self.G_matrix, x, y)
                        new_img.B_matrix[i, j] = bicubic(self.B_matrix, x, y)
                else:
                    raise ValueError("Invalid interpolation method. Choose from 'nearest', 'bilinear', or 'bicubic'.")

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix

        return new_img


    def apply_grayscale(self, in_place=False):
        new_img = SImage.new_empty_image(self.width, self.height)

        gray_matrix = SMatrix(self.R_matrix.rows, self.R_matrix.cols)

        for i in range(self.R_matrix.rows):
            for j in range(self.R_matrix.cols):
                gray_value = int((self.R_matrix[i, j] * 0.2989 + self.G_matrix[i, j] * 0.5870 + self.B_matrix[i, j] * 0.1140))
                gray_matrix[i, j] = gray_value
        
        gray_matrix = gray_matrix // 3
        new_img.R_matrix = copy.deepcopy(gray_matrix)
        new_img.G_matrix = copy.deepcopy(gray_matrix)
        new_img.B_matrix = copy.deepcopy(gray_matrix)

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix
        
        return new_img


    def apply_binary_thresholding(self, threshold, max_value=255, in_place=False): 
        new_img = SImage.new_empty_image(self.width, self.height)

        for i in range(self.height):
            for j in range(self.width):
                new_img.R_matrix[i, j] = max_value if self.R_matrix[i, j] >= threshold else 0
                new_img.G_matrix[i, j] = max_value if self.G_matrix[i, j] >= threshold else 0
                new_img.B_matrix[i, j] = max_value if self.B_matrix[i, j] >= threshold else 0

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix
        
        return new_img


    def apply_gaussian_blur(self, kernel: int, in_place=False):
        import math

        def create_gaussian_kernel(size, sigma=None):
            if size % 2 == 0:
                raise ValueError("Size must be an odd number.")

            if sigma is None:
                sigma = size / 6

            offset = size // 2
            kernel = SMatrix(size, size)

            sum_val = 0.0

            for i in range(size):
                for j in range(size):
                    x = i - offset
                    y = j - offset
                    exponent = -(x**2 + y**2) / (2 * sigma**2)
                    value = (1 / (2 * math.pi * sigma**2)) * math.exp(exponent)
                    kernel[i, j] = value
                    sum_val += value

            for i in range(size):
                for j in range(size):
                    kernel[i, j] /= sum_val

            return kernel

        kernel_mat = create_gaussian_kernel(kernel)
        new_img = SImage.new_empty_image(self.width, self.height)

        new_img.R_matrix = self.R_matrix.convolve(kernel_mat)
        new_img.G_matrix = self.G_matrix.convolve(kernel_mat)
        new_img.B_matrix = self.B_matrix.convolve(kernel_mat)

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix

        return new_img
    

    def apply_histogram_equalizer(self, in_place=False):
        new_img = SImage.new_empty_image(self.width, self.height)

        new_img.R_matrix = self._equalize_channel(self.R_matrix)
        new_img.G_matrix = self._equalize_channel(self.G_matrix)
        new_img.B_matrix = self._equalize_channel(self.B_matrix)

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix

        return new_img


    def apply_s_curve(self, function=["sigmoid", "custom"], function_params={}, in_place=False):
        lut_applied_img = self

        if function == "sigmoid":
            lut = [int(SMath.sigmoid((i / 255.0 - 0.5) * 2, function_params["shift"], function_params["steep"]) * 255) for i in range(256)]
            lut_applied_img = self.apply_lut(lut)

        elif function == "custom":
            lut = [int(SMath.custom_s_curve_function((i / 255.0 - 0.5) * 2) * 255) for i in range(256)]
            lut_applied_img = self.apply_lut(lut)

        if in_place:
            self.R_matrix = lut_applied_img.R_matrix
            self.G_matrix = lut_applied_img.G_matrix
            self.B_matrix = lut_applied_img.B_matrix

        return lut_applied_img
    

    @staticmethod
    def polar_to_cartesian(rho, theta):
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        return (x1, y1), (x2, y2)
    

    def find_road_lines(self):
        new_img = SImage.new_empty_image(self.width, self.height)
        new_img.R_matrix = copy.deepcopy(self.R_matrix)
        new_img.G_matrix = copy.deepcopy(self.G_matrix)
        new_img.B_matrix = copy.deepcopy(self.B_matrix)

        new_img.apply_grayscale(True)
        new_img.apply_gaussian_blur(5, True)
        new_img.apply_binary_thresholding(75, 255, True)
        sobel_kernel = SMatrix(3, 3)
        sobel_kernel[0, 0] = -1
        sobel_kernel[1, 0] = -2
        sobel_kernel[2, 0] = -1
        sobel_kernel[0, 1] = 0
        sobel_kernel[1, 1] = 0
        sobel_kernel[2, 1] = 0
        sobel_kernel[0, 2] = 1
        sobel_kernel[1, 2] = 2
        sobel_kernel[2, 2] = 1
        new_img.R_matrix = new_img.R_matrix.convolve(sobel_kernel)
        new_img.G_matrix = new_img.G_matrix.convolve(sobel_kernel)
        new_img.B_matrix = new_img.B_matrix.convolve(sobel_kernel)


        thetas = [math.radians(theta) for theta in range(0, 180, 1)]
        max_dist = int(math.hypot(new_img.width, new_img.height))
        
        accumulator = defaultdict(int)

        for y in range(new_img.height):
            for x in range(new_img.width):
                if new_img.R_matrix[y, x] + new_img.B_matrix[y, x]  + new_img.B_matrix[y, x] > 300:
                    for theta_idx, theta in enumerate(thetas):
                        rho = int(x * math.cos(theta) + y * math.sin(theta))
                        rho_idx = rho + max_dist
                        accumulator[(rho_idx, theta_idx)] += 1

        lines = []
        for (rho_idx, theta_idx), count in accumulator.items():
            if count >= 200:
                rho = rho_idx - max_dist
                theta = thetas[theta_idx]
                lines.append((rho, theta))

        for rho, theta in lines:
            (x1, y1), (x2, y2) = SImage.polar_to_cartesian(rho, theta)
            new_img.draw_line(x1, y1, x2, y2)


        return new_img
    

    def find_eyes(self):
        new_img = SImage.new_empty_image(self.width, self.height)
        new_img.R_matrix = copy.deepcopy(self.R_matrix)
        new_img.G_matrix = copy.deepcopy(self.G_matrix)
        new_img.B_matrix = copy.deepcopy(self.B_matrix)

        new_img.apply_grayscale(True)
        new_img.apply_gaussian_blur(5, True)
        kernel = SMatrix(3, 3)
        kernel[0, 0] = 1
        kernel[1, 0] = 1
        kernel[2, 0] = 1
        kernel[0, 1] = 1
        kernel[1, 1] = -8
        kernel[2, 1] = 1
        kernel[0, 2] = 1
        kernel[1, 2] = 1
        kernel[2, 2] = 1
        new_img.R_matrix = new_img.R_matrix.convolve(kernel)
        new_img.G_matrix = new_img.G_matrix.convolve(kernel)
        new_img.B_matrix = new_img.B_matrix.convolve(kernel)
        new_img.apply_s_curve("sigmoid", {"shift": -0.5, "steep": 5}, True)
        new_img.apply_binary_thresholding(30, 255, True)

        accumulator = defaultdict(int)
        height, width = new_img.height, new_img.width
        thetas = [math.radians(t) for t in range(0, 360, 5)]

        small_size = self.width if self.width < self.height else self.height
        
        radius_range = self.width // 40, self.width // 5
        step = (radius_range[1] - radius_range[0]) // 15

        for y in range(height):
            for x in range(width):
                if new_img.R_matrix[y, x] + new_img.G_matrix[y, x] + new_img.B_matrix[y, x] > 100:
                    for r in range(radius_range[0], radius_range[1] + 1, step):
                        for theta in thetas:
                            a = int(x - r * math.cos(theta))
                            b = int(y - r * math.sin(theta))
                            if 0 <= a < width and 0 <= b < height:
                                accumulator[(a, b, r)] += 1

        circles = []
        for (a, b, r), count in accumulator.items():
            if count >= 3 * r:
                circles.append((a, b, r))

        for a, b, r in circles:
            self.draw_circle(a, b, r)

        return self
    

    def draw_circle(self, xc, yc, r, color=255):
        def plot_circle_points(x, y):
            points = [
                (xc + x, yc + y), (xc - x, yc + y),
                (xc + x, yc - y), (xc - x, yc - y),
                (xc + y, yc + x), (xc - y, yc + x),
                (xc + y, yc - x), (xc - y, yc - x),
            ]
            for px, py in points:
                if 0 <= px < self.width and 0 <= py < self.height:
                    self.R_matrix[py, px] = color

        x = 0
        y = r
        d = 1 - r
        plot_circle_points(x, y)

        while x < y:
            x += 1
            if d < 0:
                d = d + 2 * x + 1
            else:
                y -= 1
                d = d + 2 * (x - y) + 1
            plot_circle_points(x, y)

    
    def draw_line(self, x1, y1, x2, y2, value=255):
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1

        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x2:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.R_matrix[y, x] = value
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.R_matrix[y, x] = value
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        if 0 <= x2 < self.width and 0 <= y2 < self.height:
            self.R_matrix[y2, x2] = value


    def deblur(self, factor):
        new_img = SImage.new_empty_image(self.width, self.height)
        new_img.R_matrix = copy.deepcopy(self.R_matrix)
        new_img.G_matrix = copy.deepcopy(self.G_matrix)
        new_img.B_matrix = copy.deepcopy(self.B_matrix)

        kernel = SMatrix(3, 3)
        kernel[0, 0] = 1
        kernel[1, 0] = 1
        kernel[2, 0] = 1
        kernel[0, 1] = 1
        kernel[1, 1] = -8
        kernel[2, 1] = 1
        kernel[0, 2] = 1
        kernel[1, 2] = 1
        kernel[2, 2] = 1
        temp_img = new_img.apply_s_curve("custom")
        temp_img.R_matrix = temp_img.R_matrix.convolve(kernel)
        temp_img.G_matrix = temp_img.G_matrix.convolve(kernel)
        temp_img.B_matrix = temp_img.B_matrix.convolve(kernel)

        temp_img = temp_img.apply_binary_thresholding(200)

        new_img.R_matrix += temp_img.R_matrix
        new_img.G_matrix += temp_img.G_matrix
        new_img.B_matrix += temp_img.B_matrix

        blurred_img = new_img.apply_gaussian_blur(5)

        detail_mask = SImage.new_empty_image(self.width, self.height)
        detail_mask.R_matrix = (new_img.R_matrix - blurred_img.R_matrix).multiply_by_scalar(factor, True)
        detail_mask.G_matrix = (new_img.G_matrix - blurred_img.G_matrix).multiply_by_scalar(factor, True)
        detail_mask.B_matrix = (new_img.B_matrix - blurred_img.B_matrix).multiply_by_scalar(factor, True)
        
        sharpened_img = new_img + detail_mask



        for i in range(sharpened_img.height):
            for j in range(sharpened_img.width):
                if sharpened_img.R_matrix[i, j] < 0:
                    sharpened_img.R_matrix[i, j] = 0
                elif sharpened_img.R_matrix[i, j] > 255:
                    sharpened_img.R_matrix[i, j] = 255

                if sharpened_img.G_matrix[i, j] < 0:
                    sharpened_img.G_matrix[i, j] = 0
                elif sharpened_img.G_matrix[i, j] > 255:
                    sharpened_img.G_matrix[i, j] = 255

                if sharpened_img.B_matrix[i, j] < 0:
                    sharpened_img.B_matrix[i, j] = 0
                elif sharpened_img.B_matrix[i, j] > 255:
                    sharpened_img.B_matrix[i, j] = 255
        
        return sharpened_img


    def extract_features(self):
        new_img = self.apply_gaussian_blur(5)

        for i in range(self.height):
            for j in range(self.width):
                if self.G_matrix[i, j] >= 75 and self.R_matrix[i, j] + self.B_matrix[i, j] < 75:
                    new_img.R_matrix[i, j] = 255
                    new_img.G_matrix[i, j] = 255
                    new_img.B_matrix[i, j] = 255
                else:
                    new_img.R_matrix[i, j] = 0
                    new_img.G_matrix[i, j] = 0
                    new_img.B_matrix[i, j] = 0
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        features = []

        for y in range(self.height):
            for x in range(self.width):
                if new_img.R_matrix[y, x] == 255 and not visited[y][x]:
                    queue = [(x, y)]
                    region_pixels = []
                    min_x, max_x = x, x
                    min_y, max_y = y, y
                    energy = 0
                    histogram = [0] * 256

                    while queue:
                        cx, cy = queue.pop()
                        if 0 <= cx < self.width and 0 <= cy < self.height:
                            if not visited[cy][cx] and new_img.R_matrix[cy, cx] == 255:
                                visited[cy][cx] = True
                                region_pixels.append((cx, cy))

                                min_x = min(min_x, cx)
                                max_x = max(max_x, cx)
                                min_y = min(min_y, cy)
                                max_y = max(max_y, cy)

                                val = self.G_matrix[cy, cx]
                                energy += val * val
                                histogram[val] += 1

                                queue.extend([
                                    (cx + 1, cy),
                                    (cx - 1, cy),
                                    (cx, cy + 1),
                                    (cx, cy - 1)
                                ])

                    if len(region_pixels) < 5:
                        continue

                    width = max_x - min_x + 1
                    length = max_y - min_y + 1
                    diagonal = int((width ** 2 + length ** 2) ** 0.5)
                    cx = (min_x + max_x) // 2
                    cy = (min_y + max_y) // 2

                    values = [self.G_matrix[py, px] for (px, py) in region_pixels]
                    mean = sum(values) // len(values)
                    median = sorted(values)[len(values) // 2]

                    total = sum(histogram)
                    entropy = 0
                    for count in histogram:
                        if count > 0:
                            p = count / total
                            entropy -= p * math.log2(p)

                    features.append({
                        "center": (cx, cy),
                        "width": width,
                        "length": length,
                        "diagonal": diagonal,
                        "energy": energy,
                        "entropy": entropy,
                        "mean": mean,
                        "median": median
                    })

        SImage.save_features_to_excel(features)
        return new_img


    def apply_lut(self, lut, in_place=False):
        new_img = SImage.new_empty_image(self.width, self.height)

        for i in range(self.height):
            for j in range(self.width):
                new_img.R_matrix[i, j] = lut[self.R_matrix[i, j]]
                new_img.G_matrix[i, j] = lut[self.G_matrix[i, j]]
                new_img.B_matrix[i, j] = lut[self.B_matrix[i, j]]

        if in_place:
            self.R_matrix = new_img.R_matrix
            self.G_matrix = new_img.G_matrix
            self.B_matrix = new_img.B_matrix
        
        return new_img


    @staticmethod
    def save_features_to_excel(features, filename="ozellikler.xlsx"):
        wb = Workbook()
        ws = wb.active
        ws.title = "Özellikler"

        headers = ["No", "Center", "Length", "Width", "Diagonal", "Energy", "Entropy", "Mean", "Median"]
        ws.append(headers)

        for i, f in enumerate(features, 1):
            row = [
                i,
                f"{f['center'][0]},{f['center'][1]}",
                f"{f['length']} px",
                f"{f['width']} px",
                f"{f['diagonal']} px",
                round(f["energy"], 3),
                round(f["entropy"], 3),
                f["mean"],
                f["median"]
            ]
            ws.append(row)

        wb.save(filename)
    
    
    @staticmethod
    def _equalize_channel(channel):
        height = channel.rows
        width = channel.cols

        pixels = [channel[i, j] for i in range(height) for j in range(width)]

        hist = [0] * 256
        for p in pixels:
            hist[p] += 1

        cdf = [0] * 256
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i - 1] + hist[i]

        cdf_min = next((x for x in cdf if x > 0), None)
        if cdf_min is None:
            return channel.copy()

        total_pixels = height * width
        denominator = total_pixels - cdf_min
        if denominator == 0:
            denominator = 1

        lut = [int((cdf[i] - cdf_min) / denominator * 255) for i in range(256)]
        lut = [max(0, min(255, v)) for v in lut]

        new_channel = channel.copy()
        for i in range(height):
            for j in range(width):
                new_channel[i, j] = lut[channel[i, j]]

        return new_channel
        
