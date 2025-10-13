# Python Fundamentals - 2 Study Guide

## 🎯 Learning Objectives
Master intermediate Python data structures and concepts:
- Tuples and their immutability
- Sets and set operations
- Dictionaries and key-value pairs
- String manipulation
- File handling
- Advanced list operations

---

## 📚 Table of Contents
1. [Tuples](#tuples)
2. [Sets](#sets)
3. [Dictionaries](#dictionaries)
4. [Strings](#strings)
5. [File Handling](#files)
6. [Practice Problems](#practice)

---

## 🎁 Tuples {#tuples}

### Simple Explanation (Like You're 12)
Tuples are like lists, but once you create them, you can't change them. Think of it like a sealed envelope - you can see what's inside, but you can't add or remove anything.

### Technical Definition
Tuples are **immutable, ordered sequences** that can contain elements of different types.

### Creating Tuples
```python
# With parentheses
tuple_1 = ("Max", 28, "New York")

# Without parentheses (also valid)
tuple_2 = "Linda", 25, "Miami"

# Single element tuple (note the comma!)
single = (5,)  # This is a tuple
not_tuple = (5)  # This is just an integer
```

### Key Characteristics

**1. Immutable**
```python
tuple_1 = ("Max", 28, "New York")
tuple_1[2] = "Boston"  # TypeError: cannot modify
```

**2. Ordered**
```python
person = ("Alice", 25, "New York")
name, age, city = person  # Tuple unpacking
```

**3. Heterogeneous**
```python
mixed = (1, "hello", 3.14, True)  # Different types allowed
```

### Tuple Operations

**Accessing Elements**
```python
tuple_1 = ("Max", 28, "New York")
print(tuple_1[0])   # Max
print(tuple_1[-1])  # New York
```

**Iteration**
```python
# By element
for item in tuple_1:
    print(item)

# By index
for i in range(len(tuple_1)):
    print(i, tuple_1[i])

# With enumerate
for idx, val in enumerate(tuple_1):
    print(idx, val)
```

**Tuple Methods**
```python
my_tuple = ('a', 'p', 'p', 'l', 'e')

len(my_tuple)        # 5 - length
my_tuple.count('p')  # 2 - count occurrences
my_tuple.index('l')  # 3 - find index
```

**Concatenation and Repetition**
```python
tuple1 = (1, 2)
tuple2 = tuple1 + (3,)  # (1, 2, 3)

repeated = ('a', 'b') * 3  # ('a', 'b', 'a', 'b', 'a', 'b')
```

### Type Conversions
```python
# List to tuple
my_list = [1, 2, 3]
my_tuple = tuple(my_list)

# Tuple to list
my_tuple = (1, 2, 3)
my_list = list(my_tuple)

# String to tuple
my_str = "Hello"
str_tuple = tuple(my_str)  # ('H', 'e', 'l', 'l', 'o')
```

### Memory Efficiency
```python
import sys

my_list = [0, 1, 2, 5, 10]
my_tuple = (0, 1, 2, 5, 10)

sys.getsizeof(my_list)   # 104 bytes
sys.getsizeof(my_tuple)  # 80 bytes (more efficient!)
```

### Returning Multiple Values
```python
def divide(a, b):
    quotient = a // b
    remainder = a % b
    return quotient, remainder  # Returns tuple

q, r = divide(10, 3)  # Tuple unpacking
```

### When to Use Tuples
- **Data that shouldn't change**: Coordinates, RGB colors, database records
- **Dictionary keys**: Tuples can be keys, lists cannot
- **Function returns**: Return multiple values
- **Memory efficiency**: When you have large amounts of data

---

## 🎲 Sets {#sets}

### Simple Explanation (Like You're 12)
Sets are like a bag of unique items. If you try to put two identical items in, the bag automatically removes the duplicate. Also, items in a set have no specific order.

### Technical Definition
Sets are **unordered collections of unique elements**.

### Creating Sets
```python
# Basic set
set1 = {1, 5, 10, 4, 6, 9, 19}

# Empty set (must use set())
empty_set = set()  # Correct
empty_dict = {}    # This is a dictionary!

# From list (removes duplicates)
numbers = {2, 4, 6, 6, 2, 8}  # {2, 4, 6, 8}
```

### Key Characteristics

**1. Unordered**
```python
set1 = {1, 5, 10}
# Cannot access by index
set1[0]  # TypeError: not subscriptable
```

**2. Unique Elements**
```python
numbers = {2, 4, 6, 6, 2, 8, 6}
print(numbers)  # {2, 4, 6, 8} - duplicates removed
```

**3. Heterogeneous**
```python
mixed_set = {1, 2, "a"}  # Different types allowed
```

### Set Operations

**Adding Elements**
```python
fruits = {"apple", "banana"}
fruits.add("orange")  # Add single element
```

**Removing Elements**
```python
fruits.remove("banana")    # Raises error if not found
fruits.discard("cherry")   # No error if not found
fruits.clear()             # Remove all elements
```

**Set Mathematics**

**Union** (all elements from both sets)
```python
set1 = {"apple", "banana", "orange"}
set2 = {"orange", "strawberry", "kiwi"}

all_fruits = set1.union(set2)
# or: all_fruits = set1 | set2
# Result: {'apple', 'banana', 'orange', 'strawberry', 'kiwi'}
```

**Intersection** (common elements)
```python
common = set1.intersection(set2)
# or: common = set1 & set2
# Result: {'orange'}
```

**Difference** (elements in first but not second)
```python
diff = set1.difference(set2)
# or: diff = set1 - set2
# Result: {'apple', 'banana'}
```

**Update** (add elements from another set)
```python
set1.update(set2)  # Modifies set1 in place
```

### Checking Membership
```python
fruits = {"apple", "banana", "orange"}

"apple" in fruits      # True
"cherry" in fruits     # False
"cherry" not in fruits # True
```

### When to Use Sets
- **Remove duplicates**: Convert list to set
- **Membership testing**: Fast O(1) lookup
- **Mathematical operations**: Union, intersection, difference
- **Unique collections**: When order doesn't matter

---

## 📖 Dictionaries {#dictionaries}

### Simple Explanation (Like You're 12)
Dictionaries are like real dictionaries - you look up a word (key) to find its definition (value). Each key is unique and points to a specific value.

### Technical Definition
Dictionaries are **unordered collections of key-value pairs**.

### Creating Dictionaries
```python
# Basic dictionary
person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

# Empty dictionary
empty_dict = {}
# or: empty_dict = dict()

# From tuples
pairs = [("a", 1), ("b", 2)]
dict_from_pairs = dict(pairs)
```

### Accessing Values
```python
person = {"name": "John", "age": 30}

# Using key
print(person["name"])  # John

# Using get() (safer)
print(person.get("name"))      # John
print(person.get("email"))     # None (no error)
print(person.get("email", "N/A"))  # N/A (default value)
```

### Modifying Dictionaries
```python
# Add or update
person["email"] = "john@email.com"
person["age"] = 31  # Update existing

# Update multiple
person.update({"phone": "123-456", "age": 32})

# Remove
del person["phone"]           # Delete key-value pair
removed = person.pop("email") # Remove and return value
person.clear()                # Remove all
```

### Dictionary Methods
```python
person = {"name": "John", "age": 30, "city": "NYC"}

# Get all keys
keys = person.keys()      # dict_keys(['name', 'age', 'city'])

# Get all values
values = person.values()  # dict_values(['John', 30, 'NYC'])

# Get all items (key-value pairs)
items = person.items()    # dict_items([('name', 'John'), ...])
```

### Iterating Dictionaries
```python
# Iterate keys
for key in person:
    print(key)

# Iterate values
for value in person.values():
    print(value)

# Iterate key-value pairs
for key, value in person.items():
    print(f"{key}: {value}")
```

### Dictionary Comprehension
```python
# Create dictionary from range
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Filter dictionary
person = {"name": "John", "age": 30, "city": "NYC"}
filtered = {k: v for k, v in person.items() if isinstance(v, str)}
# {'name': 'John', 'city': 'NYC'}
```

### Nested Dictionaries
```python
students = {
    "student1": {"name": "John", "grade": 85},
    "student2": {"name": "Jane", "grade": 92}
}

print(students["student1"]["name"])  # John
```

### When to Use Dictionaries
- **Key-value associations**: Phone book, configuration settings
- **Fast lookups**: O(1) average time complexity
- **Counting**: Count occurrences of items
- **Caching**: Store computed results

---

## 🔤 Strings {#strings}

### String Methods
```python
text = "Hello World"

# Case conversion
text.upper()      # "HELLO WORLD"
text.lower()      # "hello world"
text.capitalize() # "Hello world"
text.title()      # "Hello World"

# Searching
text.find("World")    # 6 (index)
text.index("World")   # 6 (raises error if not found)
text.count("l")       # 3

# Checking
text.startswith("Hello")  # True
text.endswith("World")    # True
text.isalpha()            # False (has space)
text.isdigit()            # False

# Splitting and joining
words = text.split()      # ["Hello", "World"]
joined = "-".join(words)  # "Hello-World"

# Stripping whitespace
text = "  hello  "
text.strip()   # "hello"
text.lstrip()  # "hello  "
text.rstrip()  # "  hello"

# Replacing
text.replace("World", "Python")  # "Hello Python"
```

### String Formatting
```python
name = "John"
age = 30

# f-strings (Python 3.6+)
message = f"My name is {name} and I'm {age} years old"

# format() method
message = "My name is {} and I'm {} years old".format(name, age)

# % operator (old style)
message = "My name is %s and I'm %d years old" % (name, age)
```

---

## 📁 File Handling {#files}

### Reading Files
```python
# Read entire file
with open("file.txt", "r") as file:
    content = file.read()

# Read line by line
with open("file.txt", "r") as file:
    for line in file:
        print(line.strip())

# Read all lines into list
with open("file.txt", "r") as file:
    lines = file.readlines()
```

### Writing Files
```python
# Write (overwrites existing)
with open("file.txt", "w") as file:
    file.write("Hello World\n")

# Append
with open("file.txt", "a") as file:
    file.write("New line\n")

# Write multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("file.txt", "w") as file:
    file.writelines(lines)
```

### File Modes
- `"r"`: Read (default)
- `"w"`: Write (overwrites)
- `"a"`: Append
- `"r+"`: Read and write
- `"b"`: Binary mode (e.g., "rb", "wb")

---

## 🎯 Practice Problems {#practice}

### Problem 1: Contains Duplicate (Using Set)
```python
def containsDuplicate(nums):
    return len(nums) != len(set(nums))

# Test
print(containsDuplicate([1, 2, 3, 1]))  # True
print(containsDuplicate([1, 2, 3, 4]))  # False
```

### Problem 2: Two Sum (Using Dictionary)
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

# Test
print(twoSum([2, 7, 11, 15], 9))  # [0, 1]
```

### Problem 3: Character Frequency
```python
def char_frequency(text):
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Test
print(char_frequency("hello"))  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

---

## 🔑 Key Takeaways

### Tuples
- Immutable, ordered sequences
- More memory efficient than lists
- Use for data that shouldn't change
- Can be dictionary keys

### Sets
- Unordered, unique elements
- Fast membership testing
- Mathematical set operations
- Automatically removes duplicates

### Dictionaries
- Key-value pairs
- Fast lookups by key
- Keys must be immutable
- Very versatile data structure

### Comparison Table

| Feature | List | Tuple | Set | Dictionary |
|---------|------|-------|-----|------------|
| Ordered | ✅ | ✅ | ❌ | ❌ (Python 3.7+ maintains insertion order) |
| Mutable | ✅ | ❌ | ✅ | ✅ |
| Duplicates | ✅ | ✅ | ❌ | Keys: ❌, Values: ✅ |
| Indexed | ✅ | ✅ | ❌ | By key |
| Use Case | General purpose | Immutable data | Unique items | Key-value pairs |

---

## 💡 Common Mistakes to Avoid

1. **Single element tuple**: Use `(5,)` not `(5)`
2. **Empty set**: Use `set()` not `{}` (that's a dict)
3. **Modifying tuple**: Tuples are immutable
4. **Set indexing**: Sets don't support indexing
5. **Dictionary key types**: Keys must be immutable

---

## 📦 Modules {#modules}

### Simple Explanation (Like You're 12)
Modules are like toolboxes. Instead of carrying all your tools everywhere, you keep them organized in different boxes. When you need a hammer, you open the toolbox and take it out. In Python, modules are files with useful functions you can "borrow" when needed.

### Technical Definition
Modules are Python files containing functions, classes, and variables that can be imported and reused in other programs.

### Built-in Modules
```python
# Import entire module
import math
print(math.sqrt(16))  # 4.0
print(math.pi)        # 3.141592653589793

# Import specific functions
from math import sqrt, pi
print(sqrt(16))  # 4.0

# Import with alias (nickname)
import math as m
print(m.sqrt(16))  # 4.0
```

### Common Built-in Modules
- **math**: Mathematical functions (sqrt, sin, cos, pi)
- **random**: Random numbers (random(), randint(), choice())
- **datetime**: Dates and times
- **os**: Operating system operations (file paths, directories)
- **sys**: System-specific parameters (command line arguments)
- **json**: Working with JSON data

### Creating Your Own Module
```python
# File: my_module.py
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

# File: main.py
import my_module
print(my_module.greet("Alice"))  # Hello, Alice!
```

---

## 🎨 Functional Programming {#functional}

### Simple Explanation (Like You're 12)
Functional programming is like using shortcuts. Instead of writing long instructions every time, you create small, reusable pieces that do one thing really well.

### Lambda Functions - Quick Functions

**Simple Explanation**: Lambda is like a sticky note with instructions. You write a quick function without giving it a name.

```python
# Regular function
def square(x):
    return x ** 2

# Lambda function (one-liner)
square = lambda x: x ** 2

print(square(5))  # 25
```

**When to Use Lambda**:
- Short, simple operations
- Used once or with map/filter
- Don't need a full function definition

**Examples**:
```python
# Multiple arguments
add = lambda x, y: x + y
print(add(3, 5))  # 8

# With default values
greet = lambda name="Guest": f"Hello, {name}!"
print(greet())  # Hello, Guest!
```

### map() - Transform Everything

**Simple Explanation**: Imagine you have a magic wand that transforms everything it touches. `map()` is like that wand - it applies the same transformation to every item in a list.

```python
numbers = [1, 2, 3, 4, 5]

# Square every number
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Double every number
doubled = list(map(lambda x: x*2, numbers))
print(doubled)  # [2, 4, 6, 8, 10]
```

**How it Works**:
1. Takes a function and a list
2. Applies function to each element
3. Returns transformed list

**With Multiple Lists**:
```python
list1 = [1, 2, 3]
list2 = [10, 20, 30]

# Add corresponding elements
result = list(map(lambda x, y: x + y, list1, list2))
print(result)  # [11, 22, 33]
```

### filter() - Pick the Good Ones

**Simple Explanation**: Imagine sorting candies - you only keep the ones you like. `filter()` does the same - it keeps only items that pass your test.

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Keep only even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Keep only numbers greater than 5
big_numbers = list(filter(lambda x: x > 5, numbers))
print(big_numbers)  # [6, 7, 8, 9, 10]
```

**How it Works**:
1. Takes a function that returns True/False
2. Tests each element
3. Keeps only elements where function returns True

### Combining map() and filter()
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers, then square them
result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers)))
print(result)  # [4, 16, 36, 64, 100]
```

**Visual Flow**:
```
[1,2,3,4,5,6,7,8,9,10]
    ↓ filter (keep evens)
[2,4,6,8,10]
    ↓ map (square each)
[4,16,36,64,100]
```

---

## 🔍 Scope of Variables {#scope}

### Simple Explanation (Like You're 12)
Scope is like rooms in a house. Variables created in your bedroom (function) can only be used there. But variables in the living room (global) can be seen from anywhere in the house.

### Local vs Global Variables

**Local Variables** - Live inside functions
```python
def my_function():
    local_var = "I only exist here"
    print(local_var)  # Works

my_function()
print(local_var)  # Error! Can't see it outside
```

**Global Variables** - Live everywhere
```python
global_var = "I'm everywhere"

def my_function():
    print(global_var)  # Can see global variable

my_function()  # Works
print(global_var)  # Also works
```

### Modifying Global Variables
```python
counter = 0

def increment():
    global counter  # Tell Python we want to change the global variable
    counter += 1

increment()
print(counter)  # 1

increment()
print(counter)  # 2
```

**Warning**: Using `global` is usually not recommended. Better to return values:
```python
counter = 0

def increment(count):
    return count + 1

counter = increment(counter)  # Better approach
```

### LEGB Rule - Where Python Looks for Variables

Python searches in this order:
1. **L**ocal - Inside current function
2. **E**nclosing - Inside outer functions
3. **G**lobal - Top level of file
4. **B**uilt-in - Python's built-in names

```python
x = "global"

def outer():
    x = "enclosing"
    
    def inner():
        x = "local"
        print(x)  # Prints "local"
    
    inner()
    print(x)  # Prints "enclosing"

outer()
print(x)  # Prints "global"
```

---

## 📞 Call by Value vs Call by Reference {#call}

### Simple Explanation (Like You're 12)
Imagine you have a toy. If you give someone a **photo** of your toy (call by value), they can't change your actual toy. But if you give them the **actual toy** (call by reference), they can modify it!

### Immutable Objects - Like Photos (Can't Change Original)

**Immutable types**: int, float, str, tuple

```python
def modify_number(x):
    x = x + 10
    print(f"Inside: {x}")

num = 5
modify_number(num)
print(f"Outside: {num}")

# Output:
# Inside: 15
# Outside: 5  (unchanged!)
```

**What Happened**: Function got a copy, changes don't affect original.

### Mutable Objects - Like the Real Thing (Can Change Original)

**Mutable types**: list, dict, set

```python
def modify_list(lst):
    lst.append(4)
    print(f"Inside: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"Outside: {my_list}")

# Output:
# Inside: [1, 2, 3, 4]
# Outside: [1, 2, 3, 4]  (changed!)
```

**What Happened**: Function got reference to original, changes affect it.

### The Tricky Part - Reassignment vs Modification

**Reassignment** - Creates new object (doesn't change original)
```python
def reassign_list(lst):
    lst = [10, 20, 30]  # Creates NEW list
    print(f"Inside: {lst}")

my_list = [1, 2, 3]
reassign_list(my_list)
print(f"Outside: {my_list}")  # [1, 2, 3] - unchanged!
```

**Modification** - Changes original object
```python
def modify_list(lst):
    lst.append(4)  # Modifies SAME list
    print(f"Inside: {lst}")

my_list = [1, 2, 3]
modify_list(my_list)
print(f"Outside: {my_list}")  # [1, 2, 3, 4] - changed!
```

### How to Protect Your Data
```python
def safe_modify(lst):
    # Make a copy first
    new_list = lst.copy()  # or lst[:]
    new_list.append(4)
    return new_list

my_list = [1, 2, 3]
result = safe_modify(my_list)

print(f"Original: {my_list}")  # [1, 2, 3] - safe!
print(f"New: {result}")        # [1, 2, 3, 4]
```

### Quick Reference Table

| Type | Example | Behavior | Original Changes? |
|------|---------|----------|-------------------|
| int | `5` | Call by value | ❌ No |
| str | `"hello"` | Call by value | ❌ No |
| tuple | `(1,2,3)` | Call by value | ❌ No |
| list | `[1,2,3]` | Call by reference | ✅ Yes (if modified) |
| dict | `{"a":1}` | Call by reference | ✅ Yes (if modified) |
| set | `{1,2,3}` | Call by reference | ✅ Yes (if modified) |

---

## 🎯 More Practice Problems {#more-practice}

### Problem 4: Intersection of Two Arrays (LeetCode 349)
```python
def intersection(nums1, nums2):
    # Convert to sets and find common elements
    return list(set(nums1) & set(nums2))

# Test
print(intersection([1,2,2,1], [2,2]))  # [2]
print(intersection([4,9,5], [9,4,9,8,4]))  # [9,4] or [4,9]
```

**Why Sets?**: Automatically removes duplicates and has fast lookup.

### Problem 5: Majority Element (LeetCode 169)
```python
def majorityElement(nums):
    counts = {}
    for num in nums:
        counts[num] = counts.get(num, 0) + 1
        if counts[num] > len(nums) // 2:
            return num

# Test
print(majorityElement([3,2,3]))  # 3
print(majorityElement([2,2,1,1,1,2,2]))  # 2
```

**Strategy**: Count occurrences, return when count exceeds half.

### Problem 6: Running Sum (LeetCode 1480)
```python
def runningSum(nums):
    result = []
    total = 0
    for num in nums:
        total += num
        result.append(total)
    return result

# Test
print(runningSum([1,2,3,4]))  # [1,3,6,10]
```

**Pattern**: Keep cumulative sum as you iterate.

---

## 🎓 Interview Questions & Answers

### Q1: What's the difference between a tuple and a list?
**Answer**: 
- **Tuples are immutable** (can't change after creation), **lists are mutable**
- Tuples use less memory and are faster
- Tuples can be dictionary keys, lists cannot
- Use tuples for data that shouldn't change (coordinates, RGB colors)
- Use lists for data that needs to be modified

### Q2: When should I use a set instead of a list?
**Answer**:
- When you need **unique elements only** (no duplicates)
- When you need **fast membership testing** (`x in set` is O(1))
- When you need **mathematical operations** (union, intersection, difference)
- When **order doesn't matter**

### Q3: What's the difference between `remove()` and `discard()` for sets?
**Answer**:
- `remove(x)`: Raises KeyError if element not found
- `discard(x)`: Does nothing if element not found (safe)
- Use `discard()` when you're not sure if element exists

### Q4: What's the difference between `dict[key]` and `dict.get(key)`?
**Answer**:
- `dict[key]`: Raises KeyError if key doesn't exist
- `dict.get(key)`: Returns None if key doesn't exist
- `dict.get(key, default)`: Returns default value if key doesn't exist
- Use `.get()` when key might not exist

### Q5: What's the difference between `map()` and `filter()`?
**Answer**:
- `map()`: **Transforms** each element (same number of elements)
- `filter()`: **Selects** elements based on condition (fewer or same elements)
- `map()` returns transformed values
- `filter()` returns True/False and keeps True values

### Q6: What's the difference between local and global variables?
**Answer**:
- **Local**: Defined inside function, only accessible there
- **Global**: Defined outside functions, accessible everywhere
- Local variables are destroyed after function ends
- Use `global` keyword to modify global variables (not recommended)

### Q7: Why does modifying a list inside a function change the original?
**Answer**:
- Lists are **mutable** objects
- Functions receive a **reference** to the original list
- Modifications affect the original object
- To avoid this, create a copy: `lst.copy()` or `lst[:]`

### Q8: What's the difference between reassignment and modification?
**Answer**:
- **Reassignment** (`lst = [1,2,3]`): Creates new object, doesn't affect original
- **Modification** (`lst.append(4)`): Changes original object
- Reassignment changes local reference only
- Modification changes the actual object

---

## 📝 Concise Summary

### Week 2 Core Concepts

**Data Structures**:
- **Tuples**: Immutable, ordered, memory-efficient
- **Sets**: Unique, unordered, fast lookup
- **Dictionaries**: Key-value pairs, fast access

**Functional Programming**:
- **Lambda**: Quick anonymous functions
- **map()**: Transform all elements
- **filter()**: Select elements by condition

**Scope & References**:
- **LEGB Rule**: Local → Enclosing → Global → Built-in
- **Immutable**: int, str, tuple (safe from changes)
- **Mutable**: list, dict, set (can be changed)

**Key Patterns**:
- Use sets for uniqueness and fast lookup
- Use dictionaries for counting and mapping
- Use list comprehensions over map/filter (more Pythonic)
- Create copies of mutable objects when needed

This comprehensive guide covers all Python Fundamentals - 2 concepts with clear explanations and examples!