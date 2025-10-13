# Week 4: Python for ML - Coding Guide

## 📓 Notebooks: Python_for_ML_Live_Class_Notebook.ipynb, numpy_test.ipynb

## 🎯 Topics: NumPy for Machine Learning

---

## 🔧 NumPy Basics

### Array Creation
```python
import numpy as np

# From list
arr = np.array([1, 2, 3, 4, 5])

# Special arrays
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
identity = np.eye(3)
range_arr = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)
```

### Array Operations
```python
# Arithmetic
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)  # [5 7 9]
print(a * 2)  # [2 4 6]

# Broadcasting
matrix = np.array([[1, 2], [3, 4]])
print(matrix + 10)  # Adds 10 to all elements
```

### Indexing and Slicing
```python
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[2])      # 2
print(arr[1:4])    # [1 2 3]
print(arr[::2])    # [0 2 4]

# 2D arrays
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix[0, 1])  # 2
print(matrix[:, 1])  # [2 5]
```

### Reshaping
```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()
transposed = reshaped.T
```

### Statistical Operations
```python
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))
print(np.median(arr))
print(np.std(arr))
print(np.sum(arr))
print(np.min(arr))
print(np.max(arr))
```

### File I/O
```python
# Save
np.save('array.npy', arr)
np.savetxt('array.csv', arr, delimiter=',')

# Load
loaded = np.load('array.npy')
loaded_csv = np.loadtxt('array.csv', delimiter=',')
```

---

## 💡 Key Takeaways

- NumPy is foundation for ML libraries
- Vectorized operations are fast
- Broadcasting enables efficient computation
- Essential for data manipulation

---

*Master NumPy for efficient ML workflows!*
