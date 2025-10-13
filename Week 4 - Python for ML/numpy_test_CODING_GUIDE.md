# Coding Guide: numpy_test.ipynb
*Comprehensive guide for beginners learning NumPy and Pandas basics*

---

## 📚 Overview
This notebook covers fundamental NumPy operations and basic Pandas concepts essential for Machine Learning. It includes array manipulation, mathematical operations, and data analysis techniques.

---

## 🔧 Library Imports

### Why These Libraries?
```python
import numpy as np
import pandas as pd
```

**numpy (imported as np):**
- Core library for numerical computing in Python
- Provides fast array operations (100x faster than Python lists)
- Foundation for most ML libraries (scikit-learn, TensorFlow, PyTorch)
- Enables vectorized operations (operations on entire arrays at once)

**pandas (imported as pd):**
- Built on top of NumPy for data manipulation
- Provides DataFrame structure (like Excel spreadsheets in code)
- Essential for data cleaning, transformation, and analysis
- Industry standard for data preprocessing in ML pipelines

---

## 📊 Section 1: NumPy Array Basics

### 1.1 Array Indexing and Slicing
```python
data = np.array([1,2,3])
print(data[1])      # Output: 2
print(data[0:2])    # Output: [1 2]
print(data[1:])     # Output: [2 3]
print(data[-2:])    # Output: [2 3]
```

**Key Concepts:**
- **Indexing**: Access single elements using `array[index]`
  - Index starts at 0 (first element is `data[0]`)
  - Negative indices count from end (`data[-1]` is last element)

- **Slicing**: Extract portions using `array[start:end]`
  - `start` is inclusive, `end` is exclusive
  - `data[1:]` means "from index 1 to end"
  - `data[:2]` means "from start to index 2 (exclusive)"

**Why This Matters:**
- Slicing creates views (not copies) - memory efficient
- Essential for extracting features from datasets
- Used extensively in image processing and time series analysis

### 1.2 Array Properties
```python
array_1 = np.array([1,2,3,4,5])
print(array_1.ndim)    # Output: 1 (number of dimensions)
print(array_1.shape)   # Output: (5,) (shape tuple)
print(array_1.size)    # Output: 5 (total elements)
print(array_1.dtype)   # Output: int64 (data type)
```

**Understanding Properties:**
- **ndim**: Number of dimensions (1D, 2D, 3D, etc.)
- **shape**: Tuple showing size of each dimension
  - 1D array: `(n,)` where n is length
  - 2D array: `(rows, columns)`
- **size**: Total number of elements (product of shape values)
- **dtype**: Data type of elements (int64, float64, etc.)

**Interview Tip:**
Always check array shape before operations to avoid dimension mismatch errors!

### 1.3 Multi-Dimensional Arrays
```python
array_2 = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print(array_2.ndim)    # Output: 2
print(array_2.shape)   # Output: (2, 5) - 2 rows, 5 columns
print(array_2.size)    # Output: 10
```

**2D Array Structure:**
```
Row 0: [1, 2, 3, 4, 5]
Row 1: [6, 7, 8, 9, 10]
```

**Why 2D Arrays Matter:**
- Represent matrices in linear algebra
- Store tabular data (like spreadsheets)
- Foundation for image data (height × width × channels)

### 1.4 Array Reshaping
```python
array_3 = np.array([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
array_4 = array_3.reshape(5,3)
```

**Before Reshape (3×5):**
```
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]
 [11 12 13 14 15]]
```

**After Reshape (5×3):**
```
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]
 [13 14 15]]
```

**Reshape Rules:**
- Total elements must remain same (3×5 = 5×3 = 15)
- Use `-1` for auto-calculation: `reshape(-1, 3)` auto-calculates rows
- Creates a view (not copy) when possible - memory efficient

**ML Application:**
- Flatten images for neural networks: `image.reshape(-1)` or `image.flatten()`
- Prepare data for different model architectures

---

## 🏗️ Section 2: Array Creation Methods

### 2.1 Creating Special Arrays
```python
# Array of zeros
np.zeros(10)                    # 1D array: [0. 0. 0. ... 0.]
np.zeros(10, dtype=np.int64)    # Specify data type: [0 0 0 ... 0]

# Array of ones
np.ones(5)                      # [1. 1. 1. 1. 1.]
```

**When to Use:**
- **zeros**: Initialize arrays, create masks, padding
- **ones**: Create weight matrices, normalization
- **dtype parameter**: Control memory usage and precision

**Memory Optimization:**
```python
# int8: -128 to 127 (1 byte)
# int16: -32,768 to 32,767 (2 bytes)
# int32: -2 billion to 2 billion (4 bytes)
# int64: very large numbers (8 bytes) - default
```

### 2.2 Range-Based Arrays
```python
# arange: like Python's range()
np.arange(2, 9, 2)              # [2 4 6 8] - start, stop, step

# linspace: evenly spaced numbers
np.linspace(0, 10, num=5)       # [0. 2.5 5. 7.5 10.] - includes endpoint
```

**Key Differences:**
- **arange**: Step size determines number of elements
- **linspace**: Number of elements determines step size

**Use Cases:**
- arange: Integer sequences, iteration indices
- linspace: Smooth curves, plotting, interpolation

---

## 🔢 Section 3: Array Operations

### 3.1 Sorting
```python
array_4 = np.array([2,1,5,10,19,12,8,15])
print(np.sort(array_4))         # [1 2 5 8 10 12 15 19]
```

**Sorting Methods:**
- `np.sort(array)`: Returns sorted copy
- `array.sort()`: Sorts in-place (modifies original)
- `np.argsort(array)`: Returns indices that would sort array

**Advanced Sorting:**
```python
# Sort 2D array
arr_2d = np.array([[3, 1, 2], [6, 4, 5]])
np.sort(arr_2d, axis=1)         # Sort each row
np.sort(arr_2d, axis=0)         # Sort each column
```

### 3.2 Array Concatenation
```python
x = np.array([[1,2], [3,4]])
y = np.array([[5,6]])
result = np.concatenate((x,y), axis=0)
```

**Result:**
```
[[1 2]
 [3 4]
 [5 6]]
```

**Concatenation Axes:**
- `axis=0`: Stack vertically (add rows)
- `axis=1`: Stack horizontally (add columns)

**Alternative Methods:**
```python
np.vstack([x, y])               # Vertical stack (same as axis=0)
np.hstack([x, y])               # Horizontal stack (same as axis=1)
np.concatenate([x, y], axis=0)  # General concatenation
```

### 3.3 Element-wise Operations
```python
array_5 = np.array([1,2])
array_6 = np.ones(2, dtype=int)

print(array_5 + array_6)        # [2 3] - addition
print(array_5 - array_6)        # [0 1] - subtraction
print(array_5 * array_6)        # [1 2] - multiplication
```

**Vectorization Benefits:**
- No loops needed - operations apply to all elements
- Much faster than Python loops (C-level optimization)
- Cleaner, more readable code

**Broadcasting Example:**
```python
array_8 = np.array([1,3])
print(array_8 * 3.14)           # [3.14 9.42] - scalar broadcast
```

---

## 📈 Section 4: Statistical Operations

### 4.1 Aggregation Functions
```python
array_7 = np.arange(2,20,2)     # [2 4 6 8 10 12 14 16 18]
print(array_7.sum())            # 90 - sum of all elements
print(array_7.min())            # 2 - minimum value
print(array_7.max())            # 18 - maximum value
print(array_7.mean())           # 10.0 - average
```

**Common Aggregations:**
- `sum()`: Total of all elements
- `mean()`: Average value
- `std()`: Standard deviation (spread of data)
- `var()`: Variance (std squared)
- `min()/max()`: Minimum/maximum values
- `argmin()/argmax()`: Index of min/max values

**Axis-wise Operations:**
```python
arr_2d = np.array([[1,2,3], [4,5,6]])
arr_2d.sum(axis=0)              # [5 7 9] - sum each column
arr_2d.sum(axis=1)              # [6 15] - sum each row
```

---

## 🎯 Section 5: Filtering and Boolean Indexing

### 5.1 Boolean Masks
```python
array_9 = np.array([10,12,8,4,5,1,7,22,40,6])

# Create boolean mask
mask = array_9 > 20
print(mask)                     # [False False ... True True False]

# Apply mask
print(array_9[mask])            # [22 40] - elements > 20
```

**How Boolean Indexing Works:**
1. Comparison creates boolean array (True/False for each element)
2. Boolean array used as mask to filter original array
3. Only elements where mask is True are selected

### 5.2 Complex Filtering
```python
# Multiple conditions with & (and) or | (or)
result = array_9[(array_9 >= 10) & (array_9 <= 30)]
print(result)                   # [10 12 22]
```

**Logical Operators:**
- `&`: AND (both conditions must be True)
- `|`: OR (at least one condition must be True)
- `~`: NOT (inverts boolean values)

**Important:** Use parentheses around each condition!

**ML Applications:**
- Filter outliers from datasets
- Select specific data ranges
- Create training/validation splits

---

## 💾 Section 6: File I/O Operations

### 6.1 Saving Arrays
```python
array_10 = np.array([1,2,3,4,5,6,7,8])

# Save in NumPy binary format (.npy)
np.save('array_10', array_10)

# Save as CSV text file
np.savetxt('array_10.csv', array_10)
```

**File Formats:**
- **.npy**: NumPy binary format
  - Fast to load/save
  - Preserves data types
  - Smaller file size
  - Only readable by NumPy

- **.csv**: Comma-separated values
  - Human-readable
  - Compatible with Excel, other tools
  - Larger file size
  - May lose precision

### 6.2 Loading Arrays
```python
# Load from .npy file
loaded_array = np.load('array_10.npy')

# Load from CSV
loaded_csv = np.loadtxt('array_10.csv')
```

**Best Practices:**
- Use .npy for intermediate results in ML pipelines
- Use .csv for sharing data with non-Python tools
- Always specify delimiter for CSV: `np.savetxt('file.csv', arr, delimiter=',')`

---

## 🔬 Section 7: Advanced Array Operations

### 7.1 Identity Matrix
```python
identity = np.eye(4)
```

**Output:**
```
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
```

**Identity Matrix Properties:**
- Square matrix (n×n)
- 1s on diagonal, 0s elsewhere
- Multiplying any matrix by identity returns original matrix

**ML Applications:**
- Initialize weight matrices
- Regularization in neural networks
- Linear algebra operations

### 7.2 Matrix Transpose
```python
a = np.array([[1,2], [3,4]])
B = a.T                         # Transpose
```

**Before Transpose:**
```
[[1 2]
 [3 4]]
```

**After Transpose:**
```
[[1 3]
 [2 4]]
```

**Transpose Rules:**
- Rows become columns, columns become rows
- Shape (m, n) becomes (n, m)
- `A.T.T == A` (double transpose returns original)

### 7.3 Matrix Trace
```python
arr = np.arange(1,10).reshape(3,3)
result = np.trace(arr)          # Sum of diagonal elements
```

**Matrix:**
```
[[1 2 3]
 [4 5 6]
 [7 8 9]]
```

**Trace = 1 + 5 + 9 = 15**

**Trace Properties:**
- Only defined for square matrices
- Sum of diagonal elements
- Used in linear algebra and ML optimization

### 7.4 Cross Product
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = np.cross(arr1, arr2)   # [-3 6 -3]
```

**Cross Product:**
- Produces vector perpendicular to both input vectors
- Only defined for 3D vectors
- Result magnitude = area of parallelogram formed by vectors

**Formula:**
```
[a1, a2, a3] × [b1, b2, b3] = [a2*b3 - a3*b2, a3*b1 - a1*b3, a1*b2 - a2*b1]
```

---

## 📝 Section 8: NumPy Assignments

### Assignment 1: Random Array Statistics
```python
array_1 = np.random.randint(1,100, [5,5])
print(f"Mean is: {array_1.mean()}")
print(f"Standard Deviation is: {array_1.std()}")
```

**Key Functions:**
- `np.random.randint(low, high, size)`: Random integers
  - `low`: Minimum value (inclusive)
  - `high`: Maximum value (exclusive)
  - `size`: Shape of output array

- `mean()`: Average value
- `std()`: Standard deviation (measure of spread)

**Understanding Standard Deviation:**
- Low std: Values clustered near mean
- High std: Values spread out
- Formula: sqrt(mean((x - mean)²))

### Assignment 2: Identity Matrix Operations
```python
array_1 = np.identity(5, dtype=np.int64)
array_2 = array_1 + 5
array_3 = np.random.randint(1,10, [5,1])
array_mul = np.multiply(array_2, array_3)
```

**Step-by-Step:**
1. Create 5×5 identity matrix
2. Add 5 to all elements (broadcasting)
3. Create random column vector (5×1)
4. Element-wise multiplication

**Broadcasting in Action:**
```
[[6 5 5 5 5]     [[7]      [[42 35 35 35 35]
 [5 6 5 5 5]  ×   [4]   =   [20 24 20 20 20]
 [5 5 6 5 5]      [4]        [20 20 24 20 20]
 [5 5 5 6 5]      [3]        [15 15 15 18 15]
 [5 5 5 5 6]]     [2]]       [10 10 10 10 12]]
```

### Assignment 3: Matrix Multiplication
```python
array_1 = np.random.randint(1,100, [3,3])
array_2 = np.random.randint(1,100, [3,3])
array_mul = np.multiply(array_1, array_2)
```

**Element-wise vs Matrix Multiplication:**
- `np.multiply(A, B)` or `A * B`: Element-wise (Hadamard product)
- `np.dot(A, B)` or `A @ B`: Matrix multiplication

**Element-wise Example:**
```
[[1 2]  *  [[5 6]  =  [[5  12]
 [3 4]]     [7 8]]     [21 32]]
```

### Assignment 4: Cumulative Sum
```python
array = np.random.randint(1,1000, [1000])
array_cum = np.cumsum(array)
print(array_cum[9])             # Sum of first 10 elements
print(array_cum[99])            # Sum of first 100 elements
```

**Cumulative Sum:**
- `cumsum()`: Running total
- `array_cum[i]` = sum of elements from index 0 to i

**Example:**
```
Array:     [1, 2, 3, 4, 5]
Cumsum:    [1, 3, 6, 10, 15]
```

**Applications:**
- Calculate running totals
- Prefix sums for optimization
- Time series cumulative metrics

### Assignment 5: Finding Min/Max Positions
```python
array = np.random.randint(1,100, [10,10])
index_min = array.argmin()
x_axis_min = index_min // 10
y_axis_min = index_min % 10
```

**Understanding argmin/argmax:**
- Returns flattened index of min/max value
- Convert to 2D coordinates:
  - Row = index // num_columns
  - Column = index % num_columns

**Example:**
```
Array shape: (10, 10)
Flattened index: 37
Row: 37 // 10 = 3
Column: 37 % 10 = 7
Position: (3, 7)
```

### Assignment 6: Replace Maximum Value
```python
array = np.random.rand(5,5)
array[array == array.max()] = 0.0
```

**Boolean Indexing for Replacement:**
1. Find maximum value: `array.max()`
2. Create boolean mask: `array == array.max()`
3. Assign new value to masked positions

**Use Cases:**
- Remove outliers
- Cap values at threshold
- Data normalization

### Assignment 7: Array Views vs Copies
```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = A[1:3, 0:2]                 # Creates view
B[0, 0] = 100                   # Modifies original A!
```

**Views vs Copies:**
- **View**: References same data (changes affect original)
- **Copy**: Independent data (changes don't affect original)

**Creating Copies:**
```python
B = A[1:3, 0:2].copy()          # Explicit copy
```

**Memory Implications:**
- Views: Memory efficient, but be careful of side effects
- Copies: Safe but use more memory

### Assignment 8: Count Even/Odd Numbers
```python
array = np.random.randint(1,100, 20)
even_count = array[array % 2 == 0].size
odd_count = array[array % 2 == 1].size
```

**Modulo Operator (%):**
- `x % 2 == 0`: Even numbers
- `x % 2 == 1`: Odd numbers

**Alternative Method:**
```python
even_count = np.sum(array % 2 == 0)
odd_count = np.sum(array % 2 == 1)
```

### Assignment 9: Reverse Columns
```python
A = np.arange(1, 17).reshape(4, 4)
B = A[:, ::-1]                  # Reverse column order
```

**Slicing Syntax:**
- `::` means all elements
- `::-1` means all elements in reverse
- `A[:, ::-1]` reverses columns (keeps rows same)
- `A[::-1, :]` reverses rows (keeps columns same)

### Assignment 10: Border Matrix
```python
array = np.zeros([8,8], dtype=np.int64)
array[0] = 1                    # Top row
array[-1] = 1                   # Bottom row
array[:,0] = 1                  # Left column
array[:,-1] = 1                 # Right column
```

**Result:**
```
[[1 1 1 1 1 1 1 1]
 [1 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1]
 [1 0 0 0 0 0 0 1]
 [1 1 1 1 1 1 1 1]]
```

**Indexing Tricks:**
- `array[0]`: First row
- `array[-1]`: Last row
- `array[:,0]`: First column
- `array[:,-1]`: Last column

---

## 🐼 Section 9: Pandas Basics

### 9.1 Creating DataFrames
```python
data = {
    'Title': ['Inception', 'Dunkirk', 'Interstellar', 'The Prestige', 'Memento'],
    'Director': ['Christopher Nolan'] * 5,
    'Rating': [8.8, 7.9, 8.6, 8.5, 8.4]
}
df = pd.DataFrame(data)
```

**DataFrame Structure:**
- Dictionary keys become column names
- Dictionary values become column data
- Automatically creates row indices (0, 1, 2, ...)

**Accessing Data:**
```python
df['Rating'].mean()             # Average rating
df[df['Director'] == 'Christopher Nolan']  # Filter rows
```

### 9.2 Filtering and Sorting
```python
data = {
    'Product': ['Laptop', 'Desktop', 'Tablet', 'Phone', 'Smartwatch'],
    'Price': [25000, 12000, 8000, 22000, 5000]
}
df = pd.DataFrame(data)

# Filter and sort
result = df[df['Price'] >= 20000].sort_values(by='Price', ascending=False)
```

**Chaining Operations:**
1. Filter: `df[df['Price'] >= 20000]`
2. Sort: `.sort_values(by='Price', ascending=False)`

**Result:**
```
  Product  Price
0  Laptop  25000
3   Phone  22000
```

### 9.3 GroupBy and Aggregation
```python
data = {
    'Store': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Item': ['Apple', 'Banana', 'Orange', 'Grape', 'Apple', 'Banana', 'Orange', 'Grape'],
    'Price': [50, 20, 30, 60, 55, 22, 33, 65],
    'Quantity': [10, 12, 15, 16, 20, 25, 30, 35]
}
df = pd.DataFrame(data)
df['Revenue'] = df['Price'] * df['Quantity']

store_df = df.groupby(by='Store').agg({
    'Price': 'mean',
    'Revenue': 'sum'
})
```

**GroupBy Process:**
1. Split data by 'Store' column
2. Apply aggregation functions:
   - Mean of 'Price'
   - Sum of 'Revenue'
3. Combine results

**Aggregation Functions:**
- `'mean'`: Average
- `'sum'`: Total
- `'count'`: Number of items
- `'min'/'max'`: Minimum/maximum
- `'std'`: Standard deviation

### 9.4 Adding Calculated Columns
```python
df['Total Expenditure'] = df['Price'] * df['Quantity']
```

**Column Operations:**
- Create new columns by assigning to `df['new_column']`
- Can use arithmetic operations on existing columns
- Vectorized operations (fast!)

---

## 🎯 Key Takeaways

### NumPy Essentials:
1. **Arrays are faster than lists** for numerical operations
2. **Vectorization** eliminates need for loops
3. **Broadcasting** enables operations on different shapes
4. **Boolean indexing** for powerful filtering
5. **Views vs copies** - understand memory implications

### Pandas Essentials:
1. **DataFrames** are like Excel spreadsheets in code
2. **Filtering** with boolean conditions
3. **GroupBy** for aggregating data
4. **Method chaining** for clean, readable code

### Best Practices:
1. Always check array shapes before operations
2. Use appropriate data types to save memory
3. Prefer vectorized operations over loops
4. Use `.copy()` when you need independent data
5. Document your code with comments

---

## 🚀 Next Steps

1. Practice with real datasets
2. Learn about advanced indexing
3. Explore pandas merge/join operations
4. Study matplotlib for visualization
5. Apply to machine learning projects

---

## 📚 Additional Resources

- NumPy Documentation: https://numpy.org/doc/
- Pandas Documentation: https://pandas.pydata.org/docs/
- NumPy Tutorial: https://numpy.org/doc/stable/user/quickstart.html
- Pandas Tutorial: https://pandas.pydata.org/docs/getting_started/intro_tutorials/

---

*Remember: Practice is key! Try modifying the code examples and experiment with different parameters.*
