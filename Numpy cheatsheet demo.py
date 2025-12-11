 import numpy as np

# 1. Array Creation
a = np.array([1, 2, 3])        # Create 1D array
b = np.array([[1, 2], [3, 4]]) # Create 2D array
z = np.zeros((2, 3))           # Array of zeros
o = np.ones((3, 3))            # Array of ones
r = np.arange(0, 10, 2)        # Even numbers 0–8
l = np.linspace(0, 1, 5)       # 5 values between 0–1

# Printing outputs
print("1D Array a:", a)
print("2D Array b:\n", b)
print("Zeros Array z:\n", z)
print("Ones Array o:\n", o)
print("Even numbers r:", r)
print("Linspace values l:", 
# numpy_cheatsheet_demo.py
import numpy as np

def section(title):
    print("\n" + "="*8, title, "="*8)

# 1. Create arrays
section("Create arrays")
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])
print("a:", a)
print("b:\n", b)

# 2. Save and load an array
section("Save / Load")
np.save('array_file.npy', a)           # saves to array_file.npy (binary .npy)
loaded = np.load('array_file.npy')     # load array back
print("Saved 'a' and loaded:", loaded)

# 3. Useful functions
section("Useful functions")
print("sum(a) =", np.sum(a))           # Sum of all elements
print("mean(a) =", np.mean(a))         # Average
print("max(a) =", np.max(a))           # Maximum value
print("sqrt(a) =", np.sqrt(a))         # Square root of each element

# 4. Reshaping arrays
section("Reshaping arrays")
x = np.arange(6).reshape(2, 3)         # Convert 1D 0..5 into 2x3 array
print("x:\n", x)
print("x transpose:\n", x.T)           # Transpose rows <> columns

# 5. Random numbers
section("Random numbers")
rand_arr = np.random.rand(2, 3)        # Random floats in [0,1)
rand_ints = np.random.randint(0, 10, size=(3,))  # Random ints 0..9
print("rand_arr:\n", rand_arr)
print("rand_ints:", rand_ints)

# 6. Linear algebra (inverse, determinant)
section("Linear algebra")
m = np.array([[1, 2], [3, 4]])
print("m:\n", m)
try:
    inv_m = np.linalg.inv(m)
    print("inverse of m:\n", inv_m)
except np.linalg.LinAlgError:
    print("m is singular; inverse does not exist.")
det_m = np.linalg.det(m)
print("determinant of m:", det_m)

# 7. Logical & sorting
section("Logical & Sorting")
arr = np.array([1, 3, 2, 3])
print("arr:", arr)
print("unique(arr):", np.unique(arr))  # Unique elements
print("sorted arr:", np.sort(arr))     # Sorted array
print("indices where arr > 2:", np.where(arr > 2))  # Indices where condition true

# 8. Matrix operations (dot / elementwise)
section("Matrix operations")
A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 2]])
print("A dot B:\n", np.dot(A, B))      # matrix multiplication
print("A * B (elementwise):\n", A * B)

# 9. More useful slicing examples
section("Slicing & indexing")
c = np.arange(1, 13).reshape(3, 4)
print("c:\n", c)
print("first row:", c[0])
print("last column:", c[:, -1])
print("submatrix (rows 0..1, cols 1..2):\n", c[0:2, 1:3])

# 10. Demonstrate boolean masking
section("Boolean masking")
mask = c % 2 == 0
print("c:\n", c)
print("mask (even numbers):\n", mask)
print("elements in c that are even:", c[mask])

print("\nDone.")
