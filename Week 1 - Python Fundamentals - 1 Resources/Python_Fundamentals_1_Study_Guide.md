# Python Fundamentals - 1 Study Guide

## 🎯 Learning Objectives
Master the foundational concepts of Python programming including:
- Basic syntax and comments
- Variables and data types
- Operators (arithmetic, comparison, logical)
- Control flow (if-else statements)
- Loops (for and while)
- Functions
- Lists and list operations

---

## 📚 Table of Contents
1. [Python Basics](#basics)
2. [Variables & Data Types](#variables)
3. [Operators](#operators)
4. [Control Flow](#control-flow)
5. [Loops](#loops)
6. [Functions](#functions)
7. [Lists](#lists)
8. [Practice Problems](#practice)

---

## 🌟 Python Basics {#basics}

### Comments (Explained Like You're 12)
Comments are like notes you write to yourself or others. The computer ignores them - they're just for humans to read!

**Technical**:
```python
# This is a single-line comment

"""
This is a multi-line comment
You can write multiple lines here
"""
```

### Print Function
The `print()` function displays output to the screen.

```python
print("Hello World!")  # Displays: Hello World!
print("Hello", "World")  # Displays: Hello World
print("a", end=' ')  # Changes ending from newline to space
print("b")  # Displays: a b
```

### Input Function
The `input()` function gets data from the user.

```python
name = input("Enter your name: ")  # Always returns a string
age = int(input("Enter your age: "))  # Convert to integer
```

---

## 💾 Variables & Data Types {#variables}

### Simple Explanation
Variables are like labeled boxes where you store information. The label is the variable name, and what's inside is the value.

### Data Types

**String (str)**: Text data
```python
name = "Matt"
print(type(name))  # <class 'str'>
```

**Integer (int)**: Whole numbers
```python
age = 20
print(type(age))  # <class 'int'>
```

**Float**: Decimal numbers
```python
gpa = 3.7
print(type(gpa))  # <class 'float'>
```

**Boolean (bool)**: True or False
```python
is_married = True
print(type(is_married))  # <class 'bool'>
```

### Variable Naming Rules
- Can contain letters, numbers, underscores
- Cannot start with a number
- Case-sensitive (age ≠ Age)
- Cannot use Python keywords

```python
# Valid
_name = "John"
age1 = 25

# Invalid
3d_num = 120  # SyntaxError: cannot start with number
```

### Type Conversion

**Explicit Typecasting**: You manually convert
```python
age = "25"
int_age = int(age)  # Convert string to integer
```

**Implicit Typecasting**: Python automatically converts
```python
int_val = 5
float_val = 2.0
result = int_val + float_val  # Result is 7.0 (float)
```

---

## ➕ Operators {#operators}

### Arithmetic Operators
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

### Assignment Operators
```python
a = 5
a += 3  # Same as: a = a + 3
a -= 2  # Same as: a = a - 2
a *= 4  # Same as: a = a * 4
a /= 2  # Same as: a = a / 2
```

### Comparison Operators
```python
a = 13
b = 20

print(a < b)   # Less than: True
print(a > b)   # Greater than: False
print(a == b)  # Equal to: False
print(a != b)  # Not equal to: True
print(a <= b)  # Less than or equal: True
print(a >= b)  # Greater than or equal: False
```

### Identity Operators
```python
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)    # True (same values)
print(a is b)    # False (different objects in memory)
print(a is c)    # True (same object)
```

### Logical Operators
```python
a = True
b = False

print(a and b)  # False (both must be True)
print(a or b)   # True (at least one is True)
print(not a)    # False (opposite of a)
```

---

## 🔀 Control Flow {#control-flow}

### If-Else Statements

**Simple Explanation**: Like a fork in the road - you go one way or another based on a condition.

```python
age = 25
threshold = 18

if age > threshold:
    print("You're eligible for voting")
else:
    print("You're NOT eligible for voting")
```

### Multiple Conditions (elif)
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

### Nested If Statements
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

### Ternary Operator (One-line if-else)
```python
num = 14
condition = (num % 2 == 0)

result = "A" if condition else "B"
print(result)  # A
```

---

## 🔁 Loops {#loops}

### For Loop

**Simple Explanation**: Repeat something a specific number of times.

```python
# Basic range
for i in range(5):
    print(i)  # Prints: 0, 1, 2, 3, 4

# Range with start and end
for i in range(10, 15):
    print(i)  # Prints: 10, 11, 12, 13, 14

# Range with step
for i in range(10, 15, 2):
    print(i)  # Prints: 10, 12, 14
```

### While Loop

**Simple Explanation**: Keep repeating while a condition is true.

```python
count = 1

while count <= 5:
    print("Number " + str(count))
    count += 1
```

### Loop Control

**break**: Exit the loop immediately
```python
for i in range(5):
    if i == 3:
        break
    print(i)  # Prints: 0, 1, 2
```

**continue**: Skip to next iteration
```python
for i in range(5):
    if i == 3:
        continue
    print(i)  # Prints: 0, 1, 2, 4
```

---

## 🔧 Functions {#functions}

### Simple Explanation
Functions are like recipes - you give them ingredients (parameters) and they give you a dish (return value).

### Basic Function
```python
def greet():
    print("Hello")

greet()  # Call the function
```

### Function with Parameters
```python
def greet_someone(name):
    print("Hello " + name)

greet_someone("John")  # Hello John
```

### Function with Return Value
```python
def add_numbers(a, b):
    return a + b

result = add_numbers(4, 8)
print(result)  # 12
```

### Default Parameters
```python
def add_numbers(a=7, b=8):
    return a + b

print(add_numbers())      # 15 (uses defaults)
print(add_numbers(2, 3))  # 5 (uses provided values)
print(add_numbers(a=2))   # 10 (a=2, b=8)
```

### Type Hints (Optional but recommended)
```python
def add_numbers(a: int, b: int) -> int:
    return a + b
```

---

## 📋 Lists {#lists}

### Simple Explanation
Lists are like shopping lists - you can add items, remove items, and check what's on the list.

### Creating Lists
```python
# Lists can contain different types
my_list = [1, 2, 3, "John", True, 'a', 'b']

# Empty list
empty_list = []
```

### Accessing Elements
```python
my_list = [1, 2, 3, 4, 5]

print(my_list[0])   # First element: 1
print(my_list[-1])  # Last element: 5
print(my_list[2])   # Third element: 3
```

### Modifying Lists
```python
my_list[0] = "new"  # Change first element
```

### List Operations
```python
# Concatenation
new_list = [0] + my_list + ['c']

# Multiplication
zeros = [0] * 10  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

### List Methods
```python
fruits = ["Apple", "Banana", "Cherry"]

fruits.append("Orange")              # Add to end
fruits.extend(["Kiwi", "Blueberry"]) # Add multiple
fruits.remove("Cherry")              # Remove by value
removed = fruits.pop()               # Remove and return last
removed = fruits.pop(2)              # Remove at index
idx = fruits.index("Banana")         # Find index
count = fruits.count("Apple")        # Count occurrences
fruits.clear()                       # Remove all
```

### Numerical List Methods
```python
nums = [2, 0, -1, 4, -2, 10]

nums.sort()              # Sort ascending
nums.sort(reverse=True)  # Sort descending
nums.reverse()           # Reverse order
nums2 = nums.copy()      # Create copy
del nums[1:3]            # Delete slice
```

### Iterating Lists
```python
# Element-wise
for num in nums:
    print(num)

# With index
for i in range(len(nums)):
    print(i, nums[i])

# With enumerate
for i, val in enumerate(nums):
    print(i, val)
```

### List Comprehensions
```python
# Create list of squares
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# Filter even numbers
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

### List Slicing
```python
my_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(my_list[1:5])      # [1, 2, 3, 4]
print(my_list[4:])       # [4, 5, 6, 7, 8, 9]
print(my_list[:4])       # [0, 1, 2, 3]
print(my_list[1:9:2])    # [1, 3, 5, 7]
print(my_list[::2])      # [0, 2, 4, 6, 8]
print(my_list[::-1])     # Reverse: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
```

### Checking Existence
```python
nums = [10, -1, -2]

print(10 in nums)  # True
print(5 in nums)   # False
```

---

## 🎯 Practice Problems {#practice}

### Problem 1: Sum of Digits
Find the sum of digits of a given number N.

```python
def find_sum_of_digits(num):
    res = 0
    while num > 0:
        remainder = num % 10
        res += remainder
        num //= 10
    return res

print(find_sum_of_digits(5013))  # Output: 9
```

### Problem 2: First Pair of Divisors
Find the first pair of non-negligible divisors for a given number N.

```python
N = 64
found_divisor = False

for i in range(2, N):
    if found_divisor:
        break
    
    for j in range(i, N):
        if i * j == N:
            ans = (i, j)
            found_divisor = True
            break

print(ans)  # Output: (2, 32)
```

### Problem 3: Two Sum (LeetCode)
Given an array and a target, return indices of two numbers that add up to target.

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
print(twoSum(list1, 9))  # Output: [0, 1]
```

### Problem 4: Contains Duplicate (LeetCode)
Return true if any value appears at least twice in the array.

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

# Test
list1 = [1, 2, 3, 1]
print(containsDuplicate(list1))  # Output: True
```

---

## 🔑 Key Takeaways

1. **Variables**: Containers for storing data
2. **Data Types**: int, float, str, bool
3. **Operators**: Arithmetic, comparison, logical
4. **Control Flow**: if-else for decision making
5. **Loops**: for and while for repetition
6. **Functions**: Reusable blocks of code
7. **Lists**: Ordered, mutable collections
8. **List Comprehensions**: Concise way to create lists

---

## 💡 Common Mistakes to Avoid

1. **Indentation**: Python uses indentation for code blocks
2. **Variable naming**: Cannot start with numbers
3. **Type errors**: Cannot add string and integer directly
4. **Index errors**: List indices start at 0
5. **Infinite loops**: Always ensure loop condition will eventually be false

---

## 📖 Additional Resources

- Python Official Documentation
- LeetCode for practice problems
- HackerRank Python track
- Real Python tutorials

This guide covers all fundamental Python concepts needed to start your programming journey!