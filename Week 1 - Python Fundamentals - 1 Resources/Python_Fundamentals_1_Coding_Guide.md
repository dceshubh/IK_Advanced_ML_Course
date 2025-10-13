# Python Fundamentals - 1 Coding Guide

## Overview
This notebook covers the essential building blocks of Python programming. Each section introduces fundamental concepts with practical examples and exercises.

---

## Step-by-Step Code Analysis

### Step 1: Comments and Print Function

```python
# This is a single-line comment

"""
This is a multi-line comment
And this is the second line
"""
```

**Purpose**: Comments document code for human readers. Python ignores them during execution.

**Key Points**:
- `#` for single-line comments
- `"""` or `'''` for multi-line comments (also called docstrings)

```python
print("Hello World!\n")
print("How are you?")
```

**Purpose**: Display output to the console.

**Arguments**:
- `end`: What to print at the end (default: `\n` newline)
- `sep`: Separator between multiple arguments (default: space)

```python
print('a', end=' ')  # Prints 'a ' without newline
print('b')           # Prints 'b' with newline
# Output: a b
```

### Step 2: Input Function

```python
name = input("Enter your name: ")
```

**Purpose**: Get user input from the console.

**Key Points**:
- Always returns a **string**
- Must convert to other types explicitly

```python
age = int(input("Enter your age: "))  # Convert to integer
```

---

### Step 3: Variables and Data Types

#### String (str)
```python
_name = "Matt"
print(_name)
print(type(_name))  # <class 'str'>
```

**Key Points**:
- Text data enclosed in quotes (" " or ' ')
- Variable names can start with underscore
- `type()` function shows the data type

#### Integer (int)
```python
age1 = 20
print(age1)
print(type(age1))  # <class 'int'>
```

**Key Points**:
- Whole numbers (positive or negative)
- No decimal point

#### Boolean (bool)
```python
is_married = True
print(is_married)
print(type(is_married))  # <class 'bool'>
```

**Key Points**:
- Only two values: `True` or `False`
- Note the capital T and F

#### Float
```python
gpa = 3.7
print(gpa)
print(type(gpa))  # <class 'float'>
```

**Key Points**:
- Numbers with decimal points
- More memory than integers

#### Variable Reassignment
```python
age = 30
print(type(age))  # <class 'int'>

age = "30"
print(type(age))  # <class 'str'>
```

**Key Points**:
- Variables can be reassigned to different types
- Python is dynamically typed

#### Variable Naming Rules
```python
# Valid
_name = "John"
age1 = 25

# Invalid
3d_num = 120  # SyntaxError: cannot start with number
```

**Rules**:
- Must start with letter or underscore
- Can contain letters, numbers, underscores
- Case-sensitive
- Cannot use Python keywords

---

### Step 4: Type Conversion

#### Explicit Typecasting
```python
age = "25"
int_age = int(age)

print(type(age))      # <class 'str'>
print(type(int_age))  # <class 'int'>
```

**Purpose**: Manually convert between types.

**Common Conversions**:
- `int()`: Convert to integer
- `float()`: Convert to float
- `str()`: Convert to string
- `bool()`: Convert to boolean

```python
age = 25
str_age = str(age)

print(type(age))      # <class 'int'>
print(type(str_age))  # <class 'str'>
```

**Error Handling**:
```python
age = "Twenty-five"
int_age = int(age)  # ValueError: invalid literal for int()
```

**Key Point**: Cannot convert non-numeric strings to numbers.

#### Implicit Typecasting
```python
int_val = 5
float_val = 2.0

print(int_val + float_val)       # 7.0
print(type(int_val + float_val)) # <class 'float'>
```

**Purpose**: Python automatically converts to prevent data loss.

**Rule**: When mixing int and float, result is float.

#### Reading Numbers from User
```python
num = int(input("Enter your number: "))
print(num)
print(type(num))  # <class 'int'>
```

**Error Handling**:
```python
try:
    num = float(input("Enter a float number: "))
except:
    print("Invalid input. Please enter a valid number")
```

---

### Step 5: Operators

#### Arithmetic Operators
```python
a = 9
b = 4

print(a + b)   # Addition: 13
print(a - b)   # Subtraction: 5
print(a * b)   # Multiplication: 36
print(a / b)   # Division: 2.25
print(a // b)  # Floor division: 2
print(a % b)   # Modulus (remainder): 1
print(a ** b)  # Exponentiation: 6561
```

**Key Operators**:
- `/`: Regular division (returns float)
- `//`: Floor division (returns integer)
- `%`: Modulus (remainder)
- `**`: Exponentiation (power)

#### Assignment Operators
```python
a = 5
b = 10

b += a  # Same as: b = b + a
print(b)  # 15

b += 1  # Same as: b = b + 1
print(b)  # 16

b *= 2  # Same as: b = b * 2
print(b)  # 32
```

**Available Operators**: `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`

#### Comparison Operators
```python
a = 13
b = 20

print(a < b)   # Less than: True
print(a == b)  # Equal to: False
print(a != b)  # Not equal to: True
print(a >= b)  # Greater than or equal: False
print(a <= b)  # Less than or equal: True
```

**Result**: Always returns boolean (True/False)

#### Identity Operators
```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

result1 = (a != b)    # False (same values)
result2 = (a is not b) # True (different objects)
result3 = (c is not a) # False (same object)

print(id(a))  # Memory address of a
print(id(b))  # Different memory address
print(id(c))  # Same as a
```

**Key Difference**:
- `==`: Compares values
- `is`: Compares object identity (memory location)

#### Logical Operators
```python
a = True
b = False

print(a and b)  # False (both must be True)
print(a or b)   # True (at least one is True)
print(not a)    # False (opposite of a)
```

**Truth Tables**:
- `and`: True only if both are True
- `or`: True if at least one is True
- `not`: Reverses the boolean value

---

### Step 6: If-Else Statements

#### Basic If-Else
```python
age = 25
threshold = 18

if age > threshold:
    print("You're eligible for voting")
else:
    print("You're NOT eligible for voting")
```

**Key Points**:
- Colon (`:`) after condition
- Indentation defines code blocks
- `else` is optional

#### Checking Even/Odd
```python
a = 8

if a % 2 == 0:
    print("Even")
else:
    print("Odd")
```

**Logic**: If remainder when divided by 2 is 0, number is even.

#### Multiple Conditions (elif)
```python
age = 25
has_license = True

if (age >= 18) and (has_license):
    print("Driving Allowed!")
elif (age >= 18) and (not has_license):
    print("Need License!")
else:
    print("Underaged!")
```

**Key Points**:
- `elif`: "else if" - checked if previous conditions are False
- Can have multiple `elif` statements
- Only one block executes

#### Nested If Statements
```python
num = 14

if num >= 10:
    if num % 2 == 0:
        print("Checkpoint A")
    else:
        print("Checkpoint B")
else:
    if num % 2 == 0:
        print("Checkpoint C")
    else:
        print("Checkpoint D")
```

**Purpose**: Handle complex decision trees.

#### Ternary Operator
```python
num = 14
condition = (num % 2 == 0)

result = "A" if condition else "B"
print(result)  # A
```

**Syntax**: `value_if_true if condition else value_if_false`

---

### Step 7: Loops

#### For Loop with range()
```python
for i in range(5):
    print(i)  # Prints: 0, 1, 2, 3, 4
```

**range() Parameters**:
- `range(stop)`: 0 to stop-1
- `range(start, stop)`: start to stop-1
- `range(start, stop, step)`: start to stop-1 with step

```python
for i in range(10, 15):
    print(i)  # Prints: 10, 11, 12, 13, 14

for i in range(10, 15, 2):
    print(i)  # Prints: 10, 12, 14
```

#### Sum Calculation
```python
SUM = 0
for x in range(1, 5):
    SUM += x
print(SUM)  # 10 (1+2+3+4)
```

#### String Concatenation in Loop
```python
message = ""
for num in range(1, 5):
    message += str(num)
print(message)  # "1234"
```

#### While Loop
```python
count = 1

while (count <= 5):
    print("Number " + str(count))
    count += 1

print("\nCount =", count)  # Count = 6
```

**Key Points**:
- Condition checked before each iteration
- Must update condition variable to avoid infinite loop

#### Loop Control: break
```python
for i in range(5):
    if i == 3:
        break
    print(i)  # Prints: 0, 1, 2
```

**Purpose**: Exit loop immediately when condition is met.

#### Loop Control: continue
```python
for i in range(5):
    if i == 3:
        continue
    print(i)  # Prints: 0, 1, 2, 4
```

**Purpose**: Skip current iteration and continue with next.

---

### Step 8: Functions

#### Basic Function
```python
def greet():
    print("Hello")

greet()  # Call the function
```

**Key Points**:
- `def` keyword defines function
- Parentheses `()` for parameters
- Colon `:` and indentation for body

#### Function with Parameters
```python
def greet_someone(name: str) -> None:
    print("Hello " + name)

greet_someone(name="John")
```

**Type Hints** (optional but recommended):
- `name: str`: Parameter type
- `-> None`: Return type

#### Function with Return Value
```python
def add_numbers(a: int, b: int) -> int:
    return a + 2 * b

print(add_numbers(a=4, b=8))  # 20
```

**Key Points**:
- `return` sends value back to caller
- Function execution stops at `return`

#### Default Parameters
```python
def add_numbers(a=7, b=8):
    return a + b

print(add_numbers())      # 15 (uses defaults)
print(add_numbers(2, 3))  # 5 (overrides defaults)
print(add_numbers(a=2))   # 10 (a=2, b=8)
```

**Purpose**: Provide default values for optional parameters.

#### Practical Example: Sum of Digits
```python
def find_sum_of_digits(num):
    res = 0
    while num > 0:
        remainder = num % 10
        res += remainder
        num //= 10
    return res

print(find_sum_of_digits(49213))  # 19
```

**Algorithm**:
1. Extract last digit using `% 10`
2. Add to result
3. Remove last digit using `// 10`
4. Repeat until number becomes 0

---

### Step 9: Lists

#### Creating Lists
```python
my_list = [1, 2, 3, "John", True, 'a', 'b']
```

**Key Points**:
- Lists are heterogeneous (can contain different types)
- Enclosed in square brackets `[]`
- Ordered and mutable

#### Accessing Elements
```python
print(my_list[0])                # First element: 1
print(my_list[len(my_list)-1])   # Last element: 'b'
print(my_list[-1])               # Last element: 'b'
print(my_list[-3])               # Third from end
```

**Indexing**:
- Positive: 0 to len-1
- Negative: -1 (last) to -len (first)

#### Modifying Lists
```python
my_list[0] = "new"
print(my_list)  # ['new', 2, 3, 'John', True, 'a', 'b']
```

#### List Operations
```python
# Concatenation
new_list = [0] + my_list + ['c']

# Multiplication
scores = [0] * 10  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

#### List Methods
```python
fruits = ["Apple", "Banana", "Cherry"]

fruits.append("Orange")              # Add to end
fruits.extend(["Kiwi", "Blueberry"]) # Add multiple items
fruits.remove("Cherry")              # Remove by value
removed_element = fruits.pop()       # Remove and return last
removed_element = fruits.pop(2)      # Remove at index 2
idx = fruits.index("Banana")         # Find index
num_of_apples = fruits.count("Apple") # Count occurrences
fruits.clear()                       # Remove all elements
```

#### Numerical List Methods
```python
nums = [2, 0, -1, 4, -2, 10]

nums.sort()              # Sort ascending (in-place)
nums.sort(reverse=True)  # Sort descending
nums.reverse()           # Reverse order
nums2 = nums.copy()      # Create shallow copy
del nums[1:3]            # Delete elements at indices 1 and 2
```

#### Iterating Lists
```python
# Element-wise iteration
for num in nums:
    print(num)

# Iteration with index
for i in range(len(nums)):
    print(i, nums[i])

# Using enumerate
for i, val in enumerate(nums):
    print(i, val)
```

#### Checking Existence
```python
case1 = (10 in nums)  # True if 10 is in list
case2 = (5 in nums)   # False if 5 is not in list
```

#### List Comprehensions
```python
# Create list of squares
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Filter even numbers
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

**Syntax**: `[expression for item in iterable if condition]`

#### List Slicing
```python
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(my_list[1:5])      # [1, 2, 3, 4]
print(my_list[4:])       # [4, 5, 6, 7, 8, 9]
print(my_list[:4])       # [0, 1, 2, 3]
print(my_list[1:9:2])    # [1, 3, 5, 7]
print(my_list[::])       # All elements
print(my_list[::2])      # Every 2nd element
print(my_list[::-1])     # Reverse
print(my_list[-2:-5:-1]) # [8, 7, 6]
```

**Syntax**: `list[start:end:step]`

---

### Step 10: Practice Problems

#### Problem 1: Two Sum
```python
def twoSum(nums, target):
    n = len(nums)
    for i in range(n):
        num1 = nums[i]
        for j in range(i+1, n):
            num2 = nums[j]
            if (num1 + num2 == target):
                return [i, j]

# Test
list1 = [2, 7, 11, 15]
result = twoSum(list1, 9)
print(result)  # [0, 1]
```

**Algorithm**:
- Nested loops to check all pairs
- Return indices when sum equals target
- Time Complexity: O(n²)

#### Problem 2: Contains Duplicate
```python
# Solution 1: Brute Force
def containsDuplicate(nums):
    n = len(nums)
    for i in range(n-1):
        for j in range(i+1, n):
            if nums[i] == nums[j]:
                return True
    return False

# Solution 2: Sorting
def containsDuplicate(nums):
    nums.sort()
    n = len(nums)
    for i in range(n-1):
        if nums[i] == nums[i+1]:
            return True
    return False
```

**Comparison**:
- Solution 1: O(n²) time, O(1) space
- Solution 2: O(n log n) time, O(1) space (better)

---

## Key Takeaways

1. **Comments**: Document code with `#` or `"""`
2. **Variables**: Dynamic typing, can reassign
3. **Type Conversion**: Explicit with `int()`, `str()`, etc.
4. **Operators**: Arithmetic, comparison, logical, identity
5. **Control Flow**: if-elif-else for decisions
6. **Loops**: for and while for repetition
7. **Functions**: Reusable code blocks with parameters and return values
8. **Lists**: Mutable, ordered collections with many built-in methods

This coding guide provides detailed explanations for every major concept in Python Fundamentals - 1!