import numpy as np


class SMatrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)
        self.matrix = [[0 for _ in range(cols)] for _ in range(rows)]

    @classmethod
    def from_nparray(cls, numpy_array):
        if not isinstance(numpy_array, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        if len(numpy_array.shape) != 2:
            raise ValueError("Input must be a 2D numpy array.")
        
        rows, cols = numpy_array.shape
        matrix = numpy_array.tolist()

        instance = cls(rows, cols)

        for i in range(rows):
            for j in range(cols):
                instance[i, j] = matrix[i][j]
                
        return instance

    def __add__(self, other):
        if isinstance(other, SMatrix):

            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape to add.")
            
            result = SMatrix(self.rows, self.cols)

            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self[i, j] + other[i, j]
            return result
        
        else:
            raise TypeError("Unsupported operand type for +: 'SMatrix' and '{}'".format(type(other).__name__))


    def __sub__(self, other):
        if isinstance(other, SMatrix):

            if self.shape != other.shape:
                raise ValueError("Matrices must have the same shape to subtract.")
            
            result = SMatrix(self.rows, self.cols)

            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self[i, j] - other[i, j]
            return result
        
        else:
            raise TypeError("Unsupported operand type for -: 'SMatrix' and '{}'".format(type(other).__name__))


    def __mul__(self, other):
        if isinstance(other, SMatrix):

            if self.cols != other.rows:
                raise ValueError("Number of columns in the first matrix must be equal to the number of rows in the second matrix.")
            
            result = SMatrix(self.rows, other.cols)

            for i in range(self.rows):
                for j in range(other.cols):
                    result[i, j] = sum(self[i, k] * other[k, j] for k in range(self.cols))
            return result
        else:
            raise TypeError("Unsupported operand type for *: 'SMatrix' and '{}'".format(type(other).__name__))




    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)):
            result = SMatrix(self.rows, self.cols)

            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = self[i, j] / scalar
            return result
        
        else:
            raise TypeError("Unsupported operand type for /: 'SMatrix' and '{}'".format(type(scalar).__name__))


    def __floordiv__(self, scalar):
        if isinstance(scalar, (int, float)):
            result = SMatrix(self.rows, self.cols)

            for i in range(self.rows):
                for j in range(self.cols):
                    result[i, j] = int(self[i, j] / scalar)
            return result
        
        else:
            raise TypeError("Unsupported operand type for /: 'SMatrix' and '{}'".format(type(scalar).__name__))
        

    def __str__(self):
        return "\n".join(["\t".join(map(str, row)) for row in self.matrix])


    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                raise IndexError("Index out of range.")
        else:
            raise TypeError("Invalid index type. Expected a tuple (row, col).")
        
        return self.matrix[row][col]


    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                raise IndexError("Index out of range.")
        else:
            raise TypeError("Invalid index type. Expected a tuple (row, col).")
        
        self.matrix[row][col] = value


    def __eq__(self, other):
        if isinstance(other, SMatrix):
            if self.shape != other.shape:
                return False
            
            for i in range(self.rows):
                for j in range(self.cols):
                    if self[i, j] != other[i, j]:
                        return False
            return True
        
        else:
            raise TypeError("Unsupported operand type for ==: 'SMatrix' and '{}'".format(type(other).__name__))
    

    def multiply_by_scalar(self, scalar, is_int=False):
        if is_int:
            for i in range(self.rows):
                for j in range(self.cols):
                    self[i, j] = int(self[i, j] * scalar)
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self[i, j] *= scalar
        return self

    
    def copy(self):
        new_matrix = SMatrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                new_matrix[i, j] = self[i, j]
        return new_matrix


    def convolve(self, kernel):
        if kernel.rows != kernel.cols:
            raise ValueError("Kernel must be square.")

        if kernel.rows % 2 == 0:
            raise ValueError("Kernel size must be an odd number.")

        offset = kernel.rows // 2

        padded_rows = self.rows + 2 * offset
        padded_cols = self.cols + 2 * offset

        padded = SMatrix(padded_rows, padded_cols)

        for i in range(self.rows):
            for j in range(self.cols):
                padded[i + offset, j + offset] = self[i, j]

        output = SMatrix(self.rows, self.cols)

        kernel_sum = 0
        for i in range(kernel.rows):
            for j in range(kernel.cols):
                kernel_sum += kernel[i, j]
        if kernel_sum == 0:
            kernel_sum = 1

        for i in range(self.rows):
            for j in range(self.cols):
                acc = 0.0
                for ki in range(kernel.rows):
                    for kj in range(kernel.cols):
                        ni = i + ki
                        nj = j + kj
                        acc += padded[ni, nj] * kernel[ki, kj]

                acc /= kernel_sum
                output[i, j] = int(min(max(acc, 0), 255))

        return output
