import numpy as np

# Basics
# Easiest 1D array
a1_array = np.array([1, 2, 3])
print(a1_array)
# 2D array
# basically a list within a list
b1_array = np.array([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]])
print(b1_array)

# Keep on nesting for 3-D 4-D array
c1_array = np.array([[[3, 4, 5], [1, 8, 9]], [[9, 4, 3], [4, 5, 6]]])
print(c1_array)
print(c1_array.shape)
print(c1_array)

# getting dimension of the array
print(a1_array.ndim)

# getting shape
# a1_array has only 1 dimension we checked it using ndim
# output in shape will have only 1 value and then a comma
# rows and columns(matrix)
print(a1_array.shape)
print(b1_array.shape)
# get type
# int 32 by default
print(a1_array.dtype)

# if you want to change the size explicitly.
a1_array = np.array([[1, 2, 3]], dtype='int16')  # this is a 2D array because of the extra brackets
print(a1_array)
print(a1_array.dtype)

# get size
# tells bytes
# int 32 means 4 bytes
# int 16 means 2 bytes
# int 8 means 1 bytes(1 byte contains of 8 bits)
print(a1_array.itemsize)

# get total size
# integers
print(a1_array.size)  # number of elements
print(a1_array.itemsize)  # size in bytes
print(a1_array.size * a1_array.itemsize)  # total memory used by the array
print(a1_array.nbytes)  # total memory used by the array

# floats value
# floats require more memory than integers
# Integers represent whole numbers without any fractional part, while floats represent numbers
# with both an integer and fractional part. To accurately represent decimal numbers and allow for
# a wider range of values, float data types require additional bits to store the fractional component and the exponent.
# NumPy supports various float data types, including float16, float32, float64, and float128,
print(b1_array.itemsize)
print(b1_array.dtype)

# always try to specify the datatype that you are working on
new_array = np.array([1.0, 2.0, 3.0], dtype='float32')
print(new_array.itemsize)  # itemsize gives the size in bytes
print(new_array.dtype)
print(new_array.shape)
print(new_array.size)

# Accessing, retrieve and change values in numpy
try:
    my_new_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3]], dtype='int32')
except Exception as e:
    print('NumPy arrays require homogeneous shapes, meaning all inner lists should '
          'have the same length, a ValueError is raised.')

my_new_array = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [0, 9, 8, 7, 6, 5, 4, 3], [5, 3, 2, 5, 6, 7, 7, 1]], dtype='int32')
print(my_new_array)
print(my_new_array.shape)

# getting a specific element [row, column]
# everything starts from 0
print(my_new_array[0, 5])

# you can also use negative notation like in lists
print(my_new_array[1, -5])

# get a specific row
# everything in row
print(my_new_array[0, :])

# now column

print(my_new_array[:, 2])

# this slicing works like [startindex: lastindex:stepsize]
# using step
print('-----')
print(my_new_array[0:7:2, :])
print('-----')
print(my_new_array[0, 1::2])
# in reverse
print(my_new_array[1, -1::-2])

# changing the original element
# 0th row 1st element
my_new_array[0, 1] = 10
print(my_new_array)

# for entire column
my_new_array[:, 2] = [99, 99, 99]
print(my_new_array)

# making a 3D numpy array
"""each position is represented as a tuple (i, j, k), 
where i denotes the outermost dimension (first set of brackets), 
j denotes the middle dimension (second set of brackets), 
and k denotes the innermost dimension (third set of brackets). 
The number after the colon represents the element value at that position.
For example, the position (0, 0, 0) refers to the element 1, 
the position (0, 0, 1) refers to the element 2, and so on."""
three_array = np.array([
    [[1, 2],
     [3, 4]],
    [[5, 6],
     [7, 8]]
])
print(three_array)
print(three_array.shape)
# get specific element
print(three_array[0, 0, 1])
# slicing
print(three_array[0, :, :])

# replacing
three_array[:, 1, :] = [[99, 99], [99, 99]]
print(three_array)

# Initialize all 0s matrix

a = np.zeros((2, 3))
print(a)

# All 1's matrix
b = np.ones((3, 2, 2), dtype='int16')
print(b)

# any other matrix
# shape then value
c = np.full((2, 2), 99, dtype='float32')
print(c)

# full like method
# reuse some array especially shape

m = np.full_like(my_new_array, 4)
print(m)

# random decimal numbers

print(np.random.rand(2, 3))

# random integers needs start value or it starts from 0
#  keeps changing
print(np.random.randint(4, 7, size=(2, 2)))

# identity matrix
print(np.identity(5))

new = np.ones((5, 5))
print(new)

z = np.zeros((3, 3))

z[1, 1] = 9
print(z)

new[1:4, 1:4] = z
print(new)

# Copying arrays
a = np.array([1, 2, 3])
# here it is pointing to the same object.  it did not create a new one
# if you alter a, it will alter b
# to prevent that use copy
b = a.copy()
print(b)

# mathematics in numpy

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# adds 2 to each element
print(a + 2)
# subtracting from each element
print(a - 2)
# times
print(a * 2)

# divides
print(a / 2)

b = np.array([1, 2, 3, 4, 5, 6, 7, 8])

print(a + b)

# power
print(a ** 2)

# sin, cosine
print(np.sin(a))

# Linear Algebra

# MATLAB
# this is matrix multiplication
a = np.ones((2, 3))
b = np.full((3, 2), 2)
print(a * b)  # this doesn't work here because the shape is different
# we are trying to get an answer for matrix multiplication

print(np.matmul(a, b))

# determinant of matrix
c = np.identity(3)
print(np.linalg.det(c))

# Statistics with numpy
# min mean max

stats = np.array([[1, 2, 3], [4, 5, 6]])
print('stats')
print(np.min(stats))
print(np.max(stats))
print(np.min(stats, axis=1))
# all
print(np.sum(stats))
# row
print(np.sum(stats, axis=0))

# reorganizing arrays
# dimensions are important very
before = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(before.shape)
# changing the shape

after = before.reshape((4, 2))
print(after)
# vertically stacking arrays
v1 = np.array([1, 2, 3, 4])
v2 = np.array([6, 7, 8, 9])
print(np.vstack([v1, v2]))
print(np.vstack([v1, v2, v2, v1]))

# horizontal stack
h1 = np.array([1, 2, 3, 4])
h2 = np.array([6, 7, 8, 9])
print(np.hstack((h1, h2)))

# don't want to use pandas and load a file from numpy array
three_array = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[5, 6, 7], [1, 5, 8]]
])
three_array = three_array.reshape(-1, 3)
print(three_array.shape)
print(three_array)
# np.savetxt('array.csv', three_array, delimiter=',', fmt='%d')
# default float
new_data = np.genfromtxt('array.csv', delimiter=',')
# converting it to int using astype
new_data = new_data.astype('int32')
print(new_data)
