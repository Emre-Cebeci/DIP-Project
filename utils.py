import math

class SMath:
    @staticmethod
    def sigmoid(x, shift = 0, steep = 1):
        return 1 / (1 + math.exp(-(x - shift) * steep))
    
    @staticmethod
    def sigmoid_horizontal_shift(x, shift_factor):
        return 1 / (1 + math.exp(-x + shift_factor))
    
    @staticmethod
    def sigmoid_steep(x, steep_factor):
        return 1 / (1 + math.exp(-x * steep_factor))
    
    @staticmethod
    def custom_s_curve_function(x):
        return 0.5 * (x**3) + 0.5