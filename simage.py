import numpy as np
import cv2


class SImage():
    def __init__(self, image_array: np.ndarray, color_space: str):
        self.color_space = color_space
        self.image_array = image_array
    
    @classmethod
    def from_file_path(cls, file_path: str, color_space: str):
        img = cv2.imread(file_path)
        
        if img is None:
            raise FileNotFoundError(f"Could not read the file: {file_path}")

        color_space = color_space.lower()

        if color_space == "rgb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_space == "gray":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_space == "hsv":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            raise ValueError(f"Invalid or unsupported color space: {color_space}")
        
        image_array = np.array(img)
        return cls(image_array, color_space)
            
    def change_color_space(self, to: str):
        to = to.lower()

        if to == "rgb":
            if self.color_space == "rgb":
                img = self.image_array
            elif self.color_space == "hsv":
                img = cv2.cvtColor(self.image_array, cv2.COLOR_HSV2RGB)
            elif self.color_space == "gray":
                img = cv2.cvtColor(self.image_array, cv2.COLOR_GRAY2RGB)
            
        elif to == "hsv":
            if self.color_space == "hsv":
                img = self.image_array
            elif self.color_space == "rgb":
                img = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2HSV)
            elif self.color_space == "gray":
                i = cv2.cvtColor(self.image_array, cv2.COLOR_GRAY2RGB)
                img = cv2.cvtColor(i, cv2.COLOR_RGB2HSV)
            
        elif to == "gray":
            if self.color_space == "gray":
                img = self.image_array
            elif self.color_space == "hsv":
                i = cv2.cvtColor(self.image_array, cv2.COLOR_HSV2RGB)
                img = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
            elif self.color_space == "rgb":
                img = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            
        else:
            raise Exception("Invalid or unsupported color space was given.")
        
        return SImage(img, to)

    def apply_binary_thresholding(self, threshold_value, max_value, in_place=False): 
        gray_img = self.change_color_space("gray")
        _, thresholded_image = cv2.threshold(gray_img.image_array, threshold_value, max_value, cv2.THRESH_BINARY)

        thresholded_image = np.array(thresholded_image)

        if in_place:
            self.image_array = thresholded_image
            self.color_space = "gray"
        
        return SImage(thresholded_image, "gray")

    def apply_gaussian_blur(self, kernel: tuple, in_place=False):
        blurred_img = cv2.GaussianBlur(self.change_color_space("rgb").image_array, kernel, 0)

        blurred_img = np.array(blurred_img)

        if in_place:
            self.image_array = blurred_img
        
        return SImage(blurred_img, "rgb")

    def apply_histogram_equalizer(self, in_place=False):
        equalized_img = cv2.equalizeHist(self.image_array)

        equalized_img = np.array(equalized_img)

        if in_place:
            self.image_array = equalized_img
        
        return SImage(equalized_img, self.color_space)
