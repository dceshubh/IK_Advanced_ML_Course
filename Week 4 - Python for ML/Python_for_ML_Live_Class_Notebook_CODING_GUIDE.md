# Coding Guide: Python_for_ML_Live_Class_Notebook.ipynb
*Comprehensive guide covering NumPy fundamentals, Pandas operations, and data manipulation for ML*

---

## 📚 Overview
This is the main live class notebook covering Python for Machine Learning. It includes comprehensive NumPy operations, Pandas data manipulation, and practical examples for ML workflows.

---

## 🔧 Section 1: Library Imports and Setup

### Import Statement
```python
import numpy as np
```

**Why import numpy as np?**
- **np** is the standard alias (convention followed globally)
- Makes code shorter and more readable
- Everyone in ML community uses this convention

**What NumPy provides:**
- Fast array operations (C-level performance)
- Mathematical functions (linear algebra, statistics)
- Random number generation
- File I/O for numerical data
- Foundation for pandas, scikit-learn, TensorFlow

---

## 📊 Section 2: NumPy Array Basics

### 2.1 Creating Arrays with Data Types
```python
arr1 = np.array([100, 78, 65, 45], dtype='int8')
```

**Understanding dtype (data type):**
- **int8**: 8-bit integer (-128 to 127) - 1 byte per element
- **int16**: 16-bit integer (-32,768 to 32,767) - 2 bytes
- **int32**: 32-bit integer (~-2 billion to 2 billion) - 4 bytes
- **int64**: 64-bit integer (default) - 8 bytes
- **float32**: 32-bit floating point - 4 bytes
- **float64**: 64-bit floating point (default) - 8 bytes

**Why specify dtype?**
- **Memory optimization**: int8 uses 8x less memory than int64
- **Performance**: Smaller types = faster operations
- **Precision control**: Choose based on data range

**Example:**
```python
# Memory comparison
arr_int64 = np.array([1,2,3,4,5])  # 40 bytes (5 * 8)
arr_int8 = np.array([1,2,3,4,5], dtype='int8')  # 5 bytes (5 * 1)
```

### 2.2 Array from List
```python
list_1 = [100, 78, 65, 45]
arr1 = np.array(list_1)
```

**Conversion Process:**
1. Python list → NumPy array
2. Automatic type inference (chooses int64 for integers)
3. Contiguous memory allocation (faster access)

**Key Differences:**

| Feature | Python List | NumPy Array |
|---------|-------------|-------------|
| Type | Heterogeneous (mixed types) | Homogeneous (same type) |
| Speed | Slow for math | 10-100x faster |
| Memory | More overhead | Compact storage |
| Operations | Limited | Rich mathematical functions |

### 2.3 Array Properties
```python
print(type(arr1))       # <class 'numpy.ndarray'>
print(arr1.size)        # 4 (total elements)
print(arr1.ndim)        # 1 (number of dimensions)
print(arr1.shape)       # (4,) (shape tuple)
print(arr1.dtype)       # int64 (data type)
```

**Understanding Each Property:**

**type(arr1):**
- Returns the class type
- `numpy.ndarray` = N-dimensional array
- "nd" stands for "n-dimensional"

**size:**
- Total number of elements
- For 1D: same as length
- For 2D: rows × columns
- For 3D: depth × rows × columns

**ndim:**
- Number of dimensions (axes)
- 1D array (vector): ndim = 1
- 2D array (matrix): ndim = 2
- 3D array (tensor): ndim = 3

**shape:**
- Tuple showing size of each dimension
- 1D: `(n,)` where n is length
- 2D: `(rows, columns)`
- 3D: `(depth, rows, columns)`

**dtype:**
- Data type of elements
- All elements must have same type
- Determines memory usage and precision

---

## 🎯 Section 3: Multi-Dimensional Arrays

### 3.1 Zero-Dimensional Arrays (Scalars)
```python
a = np.array(42)
print(a.size)       # 1
print(a.ndim)       # 0
print(a.shape)      # ()
```

**0D Arrays:**
- Single value (scalar)
- No dimensions
- Shape is empty tuple `()`
- Rarely used directly, but important conceptually

### 3.2 One-Dimensional Arrays (Vectors)
```python
a = np.array([42])
print(a.size)       # 1
print(a.ndim)       # 1
print(a.shape)      # (1,)
```

**1D Arrays:**
- Like a list or vector
- Shape: `(n,)` where n is length
- Most common for simple data sequences

### 3.3 Two-Dimensional Arrays (Matrices)
```python
a = np.array([[42]])
print(a.size)       # 1
print(a.ndim)       # 2
print(a.shape)      # (1, 1)
```

**2D Arrays:**
- Like a table or matrix
- Shape: `(rows, columns)`
- Used for tabular data, images (grayscale)

**Example with more elements:**
```python
matrix = np.array([[1,2,3], [4,5,6]])
# Shape: (2, 3) - 2 rows, 3 columns
# Visual:
# [[1 2 3]
#  [4 5 6]]
```

### 3.4 Creating Arrays from Different Structures

**From Tuples:**
```python
a = np.array((42, 21))
print(a.shape)      # (2,)
```
- Tuples work just like lists
- Converted to array automatically

**From Sets (Special Case):**
```python
a = np.array({42, 21})
print(a.ndim)       # 0
print(a.shape)      # ()
```
- Sets treated as single object (0D array)
- Not recommended - use lists or tuples instead

**Multiple Arguments (Error):**
```python
# This will cause an error:
a = np.array(42, 21)  # TypeError
# Correct way:
a = np.array([42, 21])  # Use list or tuple
```

---

## 🔪 Section 4: Array Slicing and Indexing

### 4.1 Basic Slicing Syntax
```python
list_2 = [1,2,3,4,5,6]
array_2 = np.array(list_2)

# Slicing: array[start:end:step]
print(array_2[1:4])     # [2 3 4] - indices 1,2,3
print(array_2[4:])      # [5 6] - from index 4 to end
print(array_2[:4])      # [1 2 3 4] - from start to index 3
print(array_2[-3:-1])   # [4 5] - negative indices
print(array_2[1:5:2])   # [2 4] - every 2nd element
print(array_2[::2])     # [1 3 5] - every 2nd element from start
```

**Slicing Rules:**
- **start**: Inclusive (included in result)
- **end**: Exclusive (not included in result)
- **step**: Increment between elements
- **Negative indices**: Count from end (-1 is last element)

**Visual Example:**
```
Array:    [1, 2, 3, 4, 5, 6]
Indices:   0  1  2  3  4  5
Negative: -6 -5 -4 -3 -2 -1

array[1:4]  → [2, 3, 4]  (indices 1, 2, 3)
array[-3:]  → [4, 5, 6]  (last 3 elements)
array[::2]  → [1, 3, 5]  (every 2nd element)
```

### 4.2 2D Array Indexing
```python
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

# Single element access
print(arr[0, 1])        # 2 (row 0, column 1)
print(arr[0][1])        # 2 (alternative syntax)

# Slicing rows and columns
print(arr[0][0:2])      # [1 2] (row 0, columns 0-1)
print(arr[0, 1:3])      # [2 3] (row 0, columns 1-2)
print(arr[1, -1])       # 10 (row 1, last column)
print(arr[1][-1])       # 10 (alternative)

# Entire row or column
print(arr[0, :])        # [1 2 3 4 5] (entire row 0)
print(arr[:, 1])        # [2 7] (entire column 1)
```

**2D Indexing Syntax:**
- `arr[row, col]`: Preferred (more efficient)
- `arr[row][col]`: Alternative (creates intermediate array)

**Colon (:) Meaning:**
- `:` alone means "all elements"
- `arr[0, :]` = "row 0, all columns"
- `arr[:, 1]` = "all rows, column 1"

---

## 🔄 Section 5: Copy vs View

### 5.1 Copy - Independent Data
```python
arr = np.array([1, 2, 3, 4, 5])
copied_arr = arr.copy()

# Modify original
arr[0] = 42

print(arr)          # [42  2  3  4  5]
print(copied_arr)   # [1 2 3 4 5] - unchanged!
```

**Copy Characteristics:**
- Creates new array in memory
- Changes to copy don't affect original
- Changes to original don't affect copy
- Uses more memory (duplicate data)
- Safer but slower

**When to use copy:**
- Need independent data
- Modifying without affecting original
- Passing to functions that modify data

### 5.2 View - Shared Data
```python
arr2 = np.array([1, 2, 3, 4, 5])
viewed_array = arr2.view()

# Modify original
arr2[0] = 42

print(arr2)         # [42  2  3  4  5]
print(viewed_array) # [42  2  3  4  5] - changed too!
```

**View Characteristics:**
- References same data in memory
- Changes to view affect original
- Changes to original affect view
- Memory efficient (no duplication)
- Faster but requires caution

**When to use view:**
- Memory optimization
- Read-only operations
- Temporary transformations

**Important Note:**
```python
# Slicing creates views by default!
arr = np.array([1, 2, 3, 4, 5])
slice_arr = arr[1:4]    # This is a view!
slice_arr[0] = 100      # Modifies original arr!

# To avoid this:
slice_arr = arr[1:4].copy()  # Explicit copy
```

---

## 🎭 Section 6: Boolean Indexing and Filtering

### 6.1 Creating Boolean Masks
```python
arr = np.array([10, 20, 30, 40, 50])

# Create boolean mask
mask = arr >= 30
print(mask)         # [False False  True  True  True]

# Apply mask
print(arr[mask])    # [30 40 50]
```

**How Boolean Indexing Works:**
1. Comparison creates boolean array (True/False)
2. Boolean array acts as filter
3. Only elements where mask is True are selected

**Visual Process:**
```
Original:  [10, 20, 30, 40, 50]
Condition: >= 30
Mask:      [F,  F,  T,  T,  T]
Result:    [30, 40, 50]
```

### 6.2 Direct Boolean Indexing
```python
# Combine creation and application
print(arr[arr >= 30])       # [30 40 50]
print(arr[arr < 30])        # [10 20]
```

**Common Comparisons:**
- `==`: Equal to
- `!=`: Not equal to
- `>`: Greater than
- `<`: Less than
- `>=`: Greater than or equal
- `<=`: Less than or equal

### 6.3 Multiple Conditions
```python
# AND condition (&)
result = arr[(arr >= 20) & (arr <= 40)]
print(result)       # [20 30 40]

# OR condition (|)
result = arr[(arr < 20) | (arr > 40)]
print(result)       # [10 50]

# NOT condition (~)
result = arr[~(arr == 30)]
print(result)       # [10 20 40 50]
```

**Logical Operators:**
- `&`: AND (both conditions must be True)
- `|`: OR (at least one condition must be True)
- `~`: NOT (inverts boolean values)

**Important:** Always use parentheses around each condition!

**Wrong:**
```python
arr[arr >= 20 & arr <= 40]  # Error!
```

**Correct:**
```python
arr[(arr >= 20) & (arr <= 40)]  # Works!
```

---

## 📁 Section 7: Array Manipulation

### 7.1 Reshaping Arrays
```python
arr = np.arange(12)         # [0 1 2 3 4 5 6 7 8 9 10 11]
reshaped = arr.reshape(3, 4)
```

**Result:**
```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

**Reshape Rules:**
- Total elements must remain same
- `reshape(3, 4)` = 3 rows × 4 columns = 12 elements
- Use `-1` for auto-calculation: `reshape(-1, 4)` auto-calculates rows

**Auto-calculation Examples:**
```python
arr.reshape(-1, 4)      # Auto: 3 rows, 4 columns
arr.reshape(3, -1)      # Auto: 3 rows, 4 columns
arr.reshape(-1)         # Flatten to 1D
```

### 7.2 Flattening Arrays
```python
arr_2d = np.array([[1,2,3], [4,5,6]])

# Method 1: flatten() - returns copy
flat1 = arr_2d.flatten()

# Method 2: ravel() - returns view (when possible)
flat2 = arr_2d.ravel()

# Method 3: reshape(-1) - returns view
flat3 = arr_2d.reshape(-1)
```

**All produce:** `[1 2 3 4 5 6]`

**Differences:**
- `flatten()`: Always returns copy
- `ravel()`: Returns view when possible (faster)
- `reshape(-1)`: Returns view (most flexible)

### 7.3 Transposing Arrays
```python
arr = np.array([[1,2,3], [4,5,6]])
transposed = arr.T
```

**Before:**
```
[[1 2 3]
 [4 5 6]]
```

**After:**
```
[[1 4]
 [2 5]
 [3 6]]
```

**Transpose Properties:**
- Rows become columns
- Columns become rows
- Shape (m, n) → (n, m)
- Returns view (not copy)

---

## 🔢 Section 8: Statistical Operations

### 8.1 Basic Statistics
```python
arr = np.array([1, 2, 3, 4, 5])

print(np.mean(arr))     # 3.0 (average)
print(np.median(arr))   # 3.0 (middle value)
print(np.std(arr))      # 1.414... (standard deviation)
print(np.sum(arr))      # 15 (total)
print(np.min(arr))      # 1 (minimum)
print(np.max(arr))      # 5 (maximum)
```

**Understanding Each Function:**

**mean():**
- Average value
- Formula: sum(elements) / count(elements)
- Sensitive to outliers

**median():**
- Middle value when sorted
- Not affected by outliers
- Better for skewed data

**std() - Standard Deviation:**
- Measure of spread/variability
- Low std: values close to mean
- High std: values spread out
- Formula: sqrt(mean((x - mean)²))

**sum():**
- Total of all elements
- Can specify axis for multi-dimensional

**min()/max():**
- Minimum/maximum values
- Use argmin()/argmax() for indices

### 8.2 Axis-wise Operations
```python
arr_2d = np.array([[1,2,3], [4,5,6]])

# Sum along different axes
print(arr_2d.sum())         # 21 (all elements)
print(arr_2d.sum(axis=0))   # [5 7 9] (sum each column)
print(arr_2d.sum(axis=1))   # [6 15] (sum each row)
```

**Understanding Axes:**
- **axis=0**: Operations along rows (result per column)
- **axis=1**: Operations along columns (result per row)
- **No axis**: Operation on entire array

**Visual:**
```
Array:
[[1 2 3]
 [4 5 6]]

axis=0 (↓):  [5 7 9]   (1+4, 2+5, 3+6)
axis=1 (→):  [6 15]    (1+2+3, 4+5+6)
```

---

## 🎲 Section 9: Random Number Generation

### 9.1 Random Arrays
```python
# Random floats between 0 and 1
random_arr = np.random.random((2, 3))

# Random integers
random_ints = np.random.randint(1, 100, size=(5, 5))

# Random from normal distribution
normal_arr = np.random.randn(1000)
```

**Random Functions:**

**random():**
- Uniform distribution [0, 1)
- All values equally likely

**randint(low, high, size):**
- Random integers
- `low`: inclusive
- `high`: exclusive
- `size`: shape of output

**randn():**
- Standard normal distribution
- Mean = 0, Std = 1
- Bell curve shape

**Setting Random Seed:**
```python
np.random.seed(42)      # Reproducible results
arr1 = np.random.random(5)
np.random.seed(42)      # Same seed
arr2 = np.random.random(5)
# arr1 == arr2 (identical)
```

---

## 💾 Section 10: File I/O

### 10.1 Saving Arrays
```python
arr = np.array([1,2,3,4,5])

# Save as .npy (NumPy binary)
np.save('array.npy', arr)

# Save as text file
np.savetxt('array.csv', arr, delimiter=',')
```

**File Formats:**

**.npy (NumPy Binary):**
- Fast to save/load
- Preserves dtype and shape
- Smaller file size
- Only readable by NumPy

**.csv (Text):**
- Human-readable
- Compatible with Excel
- Larger file size
- Universal format

### 10.2 Loading Arrays
```python
# Load from .npy
loaded = np.load('array.npy')

# Load from CSV
loaded_csv = np.loadtxt('array.csv', delimiter=',')
```

**Best Practices:**
- Use .npy for intermediate ML results
- Use .csv for sharing with non-Python tools
- Always specify delimiter for text files

---

## 🎯 Key Takeaways

1. **NumPy is essential** for ML - foundation of all numerical computing
2. **Vectorization** eliminates loops - much faster
3. **Broadcasting** enables operations on different shapes
4. **Views vs Copies** - understand memory implications
5. **Boolean indexing** - powerful data filtering
6. **Axis parameter** - crucial for multi-dimensional operations

---

*This guide covers the core concepts from the live class notebook. Practice these operations to build strong NumPy skills!*
