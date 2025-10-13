# Coding Guide: week4_test.ipynb
*Quick reference guide for NumPy, Pandas, and Matplotlib basics*

---

## 📚 Overview
This notebook contains test examples and practice code for Week 4 concepts including NumPy arrays, Pandas DataFrames, and basic Matplotlib visualization.

---

## 🔧 Section 1: Library Imports

```python
import numpy as np
import pandas as pd
```

**Why these imports?**
- **numpy**: Numerical computing, array operations
- **pandas**: Data manipulation, DataFrame operations
- Standard aliases (np, pd) used globally in ML community

---

## 📊 Section 2: NumPy Data Types

### 2.1 Specifying Data Types
```python
array_1 = np.array([1,2,3,4])                    # Default: int64
array_2 = np.array([1,2,3,4], dtype=np.int8)     # Explicit: int8
array_3 = np.array([1,2,3,4], dtype='int8')      # String format

print(type(array_1))        # <class 'numpy.ndarray'>
print(array_1.dtype)        # int64
print(array_2.dtype)        # int8
print(array_3.dtype)        # int8
```

**Data Type Formats:**
- **Object notation**: `dtype=np.int8`
- **String notation**: `dtype='int8'`
- Both work identically

**Common Data Types:**
- `int8`: -128 to 127 (1 byte)
- `int16`: -32,768 to 32,767 (2 bytes)
- `int32`: ~-2 billion to 2 billion (4 bytes)
- `int64`: Very large integers (8 bytes) - default
- `float32`: Single precision (4 bytes)
- `float64`: Double precision (8 bytes) - default

**Why specify dtype?**
- **Memory optimization**: Smaller types use less RAM
- **Performance**: Smaller types = faster operations
- **Precision control**: Match data range to type

**Example:**
```python
# Memory comparison
large_array_int64 = np.array([1,2,3,4,5] * 1000000)  # 40 MB
large_array_int8 = np.array([1,2,3,4,5] * 1000000, dtype='int8')  # 5 MB
# 8x memory savings!
```

---

## 🎭 Section 3: Arrays from Different Data Structures

### 3.1 Arrays from Tuples
```python
array_tuple = np.array((1,2,3))
print(array_tuple)          # [1 2 3]
print(array_tuple[0])       # 1
```

**Tuples vs Lists:**
- Both work for creating arrays
- Tuples are immutable (can't be changed)
- Lists are mutable (can be changed)
- NumPy converts both to arrays (which are mutable)

### 3.2 Arrays from Sets (Special Case)
```python
array_set = np.array({1,2,3})
print(array_set)            # {1, 2, 3} (set object)
print(array_set.ndim)       # 0 (zero-dimensional)
print(array_set.shape)      # () (empty tuple)
print(array_set[0])         # Error! Can't index 0D array
```

**Why Sets Behave Differently:**
- Sets are unordered collections
- NumPy treats entire set as single object
- Creates 0-dimensional array (scalar)
- **Not recommended** for array creation

**Best Practice:**
```python
# Convert set to list first
my_set = {1, 2, 3}
array_from_set = np.array(list(my_set))  # Correct way
```

---

## 🔪 Section 4: 2D Array Indexing

### 4.1 Two Ways to Index 2D Arrays
```python
array_2d = np.array([[1,2,3,4], [5,6,7,8]])

# Method 1: Double bracket notation
print(array_2d[0][1])       # 2

# Method 2: Comma notation (preferred)
print(array_2d[0, 1])       # 2
```

**Why comma notation is better:**
- More efficient (single operation)
- Cleaner syntax
- Standard in NumPy documentation
- Easier to read for multi-dimensional arrays

### 4.2 Slicing 2D Arrays
```python
# Slice with double brackets
print(array_2d[0][1:3])     # [2 3]

# Slice with comma notation
print(array_2d[0, 1:3])     # [2 3]
```

**Both methods work for slicing, but comma notation is preferred**

**Visual Example:**
```
Array:
[[1 2 3 4]
 [5 6 7 8]]

array_2d[0, 1]      → 2 (row 0, column 1)
array_2d[0, 1:3]    → [2 3] (row 0, columns 1-2)
array_2d[:, 1]      → [2 6] (all rows, column 1)
```

---

## 🐼 Section 5: Pandas Basics

### 5.1 Creating Series
```python
s1 = pd.Series([1,2,3,4,5])
print(s1)
```

**Output:**
```
0    1
1    2
2    3
3    4
4    5
dtype: int64
```

**Series Structure:**
- 1-dimensional labeled array
- Left column: index (0, 1, 2, ...)
- Right column: values
- Like a single column from Excel

### 5.2 Creating DataFrames
```python
data = {
    'city': ['Seattle', 'dallas'],
    'population': [10000, 20000]
}
df = pd.DataFrame(data)
```

**Output:**
```
      city  population
0  Seattle       10000
1   dallas       20000
```

**DataFrame Structure:**
- 2-dimensional labeled data structure
- Rows: indexed (0, 1, 2, ...)
- Columns: named ('city', 'population')
- Like an Excel spreadsheet

### 5.3 DataFrame Information Methods
```python
# Get DataFrame info
print(df.info())
```

**Output:**
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 2 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   city        2 non-null      object
 1   population  2 non-null      int64 
dtypes: int64(1), object(1)
memory usage: 164.0+ bytes
```

**What info() shows:**
- Number of rows and columns
- Column names and types
- Non-null counts (missing data detection)
- Memory usage

```python
# Get statistical summary
print(df.describe())
```

**Output:**
```
         population
count      2.000000
mean   15000.000000
std     7071.067812
min    10000.000000
25%    12500.000000
50%    15000.000000
75%    17500.000000
max    20000.000000
```

**What describe() shows:**
- count: Number of non-null values
- mean: Average
- std: Standard deviation
- min/max: Minimum/maximum values
- 25%/50%/75%: Quartiles (percentiles)

**Use Cases:**
- `info()`: Check data types, missing values
- `describe()`: Quick statistical overview
- Essential for data exploration

---

## 📊 Section 6: Matplotlib Basics

### 6.1 Simple Line Plot
```python
import matplotlib.pyplot as plt

list_1 = [1,2,3,4]
plt.plot(list_1)
```

**What this does:**
- Creates line plot
- X-axis: indices (0, 1, 2, 3)
- Y-axis: values (1, 2, 3, 4)
- Displays plot

**matplotlib.pyplot (plt):**
- Standard plotting library
- `plt.plot()`: Create line plot
- Automatically shows plot in Jupyter notebooks

### 6.2 Plot Customization Basics
```python
# Add labels and title
plt.plot(list_1)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Simple Line Plot')
plt.show()
```

**Common Plot Functions:**
- `plt.plot()`: Line plot
- `plt.scatter()`: Scatter plot
- `plt.bar()`: Bar chart
- `plt.hist()`: Histogram
- `plt.xlabel()`: X-axis label
- `plt.ylabel()`: Y-axis label
- `plt.title()`: Plot title
- `plt.show()`: Display plot

---

## 🎯 Key Concepts Summary

### NumPy Essentials:
1. **Data types** control memory and precision
2. **Tuples** work like lists for array creation
3. **Sets** create 0D arrays (avoid this)
4. **Comma notation** preferred for 2D indexing

### Pandas Essentials:
1. **Series**: 1D labeled array
2. **DataFrame**: 2D labeled data structure
3. **info()**: Check data types and missing values
4. **describe()**: Statistical summary

### Matplotlib Essentials:
1. **plt.plot()**: Create visualizations
2. **Labels and titles**: Make plots readable
3. **Essential** for data exploration

---

## 💡 Best Practices

### NumPy:
- Use appropriate data types to save memory
- Prefer comma notation for multi-dimensional indexing
- Convert sets to lists before creating arrays

### Pandas:
- Always check data with `info()` and `describe()`
- Understand difference between Series and DataFrame
- Use meaningful column names

### Matplotlib:
- Always label axes
- Add titles to plots
- Use appropriate plot types for data

---

## 🚀 Next Steps

1. Practice creating arrays with different data types
2. Experiment with Pandas DataFrames
3. Create various plot types with Matplotlib
4. Combine NumPy, Pandas, and Matplotlib in projects

---

## 📚 Quick Reference

### NumPy Data Types:
```python
np.int8, np.int16, np.int32, np.int64
np.float32, np.float64
np.bool_, np.object_
```

### Pandas Methods:
```python
df.info()           # Data types and missing values
df.describe()       # Statistical summary
df.head(n)          # First n rows
df.tail(n)          # Last n rows
df.shape            # (rows, columns)
df.columns          # Column names
```

### Matplotlib Functions:
```python
plt.plot()          # Line plot
plt.scatter()       # Scatter plot
plt.bar()           # Bar chart
plt.hist()          # Histogram
plt.xlabel()        # X-axis label
plt.ylabel()        # Y-axis label
plt.title()         # Plot title
plt.legend()        # Add legend
plt.show()          # Display plot
```

---

*This notebook provides quick examples for testing and practicing Week 4 concepts. Use it as a reference while working on larger projects!*
