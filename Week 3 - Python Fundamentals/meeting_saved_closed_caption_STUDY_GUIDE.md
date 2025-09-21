# Python Fundamentals Meeting Study Guide 📚
*Understanding Python OOP and Advanced Concepts Like a Smart 12-Year-Old*

## 🎯 What This Guide Covers
This study guide breaks down Python object-oriented programming concepts, polymorphism, sorting, and exception handling from the meeting transcript into easy-to-understand explanations, followed by technical details and interview preparation materials.

---

## 🌟 Part 1: Simple Explanations with Illustrations

### 1. What is Polymorphism in Python?
**Simple Explanation:**
Polymorphism is like having a universal remote control that works differently depending on which device you point it at!

```
🎮 Universal Remote Example:
Remote + TV = Changes TV channels
Remote + Stereo = Changes music volume  
Remote + AC = Changes temperature

Same remote (function name), different behavior based on the device (object)!
```

**Python Example:**
```python
# Same method name, different behaviors
class Car:
    def start_engine(self):
        return "Vroom! Gas engine started!"

class ElectricCar:
    def start_engine(self):
        return "Whirr! Electric motor activated!"

# Same method name, different outputs based on object type
my_car = Car()
my_tesla = ElectricCar()

print(my_car.start_engine())     # "Vroom! Gas engine started!"
print(my_tesla.start_engine())   # "Whirr! Electric motor activated!"
```

### 2. What are the Two Types of Polymorphism?
**Simple Explanation:**

**Dynamic Polymorphism (Runtime):**
Like a chameleon that changes color based on where it sits!
```
🦎 Chameleon on green leaf → turns green
🦎 Chameleon on brown branch → turns brown
(Python decides at runtime which method to use)
```

**Static Polymorphism (Compile-time):**
Like a Swiss Army knife with different tools for different jobs!
```
🔧 Swiss Army Knife:
- Use knife blade for cutting
- Use screwdriver for screws  
- Use scissors for paper
(Same tool, different functions based on what you need)
```

### 3. What is Method Resolution Order (MRO)?
**Simple Explanation:**
MRO is like a family tree search when you need help - you ask in a specific order!

```
👨‍👩‍👧‍👦 Family Help Chain:
Child needs help → Ask Mom first
Mom doesn't know → Ask Dad  
Dad doesn't know → Ask Grandma
Grandma doesn't know → Ask Grandpa

Python Object Chain:
Object needs method → Check current class
Not found → Check parent class
Not found → Check grandparent class
Not found → Check base Object class
```

### 4. What is Custom Sorting?
**Simple Explanation:**
Custom sorting is like organizing your toys in different ways depending on what you want to find!

```
🧸 Toy Organization Examples:
Sort by size: Small → Medium → Large
Sort by color: Red → Blue → Green → Yellow
Sort by type: Cars → Dolls → Blocks → Games
Sort by favorite: Most loved → Least loved

Same toys, different organization rules!
```

### 5. What is Exception Handling?
**Simple Explanation:**
Exception handling is like having a safety net when you're learning to ride a bike!

```
🚲 Bike Riding Safety:
Try: Attempt to ride the bike
Except: If you fall, get back up and try again
Finally: Always put the bike away when done

No matter what happens, you handle it gracefully!
```

---

## 🔬 Part 2: Technical Concepts

### 1. Polymorphism Implementation

#### Dynamic Polymorphism
```python
class Animal:
    def make_sound(self):
        return "Some generic animal sound"

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

class Cat(Animal):
    def make_sound(self):
        return "Meow!"

# Dynamic polymorphism in action
def animal_concert(animals):
    for animal in animals:
        print(animal.make_sound())  # Python decides at runtime

pets = [Dog(), Cat(), Dog()]
animal_concert(pets)  # Output: Woof! Meow! Woof!
```

#### Static Polymorphism (Function Overloading)
```python
# Python doesn't have true function overloading, but we can simulate it

# Method 1: Default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))           # "Hello, Alice!"
print(greet("Bob", "Hi"))       # "Hi, Bob!"

# Method 2: Variable arguments
def add(*args):
    return sum(args)

print(add(1, 2))        # 3
print(add(1, 2, 3, 4))  # 10

# Method 3: Type checking
def process_data(data):
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, list):
        return len(data)
    elif isinstance(data, int):
        return data * 2
    else:
        return "Unknown type"
```

### 2. Method Resolution Order (MRO)

#### Understanding MRO
```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Multiple inheritance
    pass

# Check MRO
print(D.__mro__)
# Output: (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
print(d.method())  # Output: "B" (follows MRO: D → B → C → A)
```

#### Using super() with MRO
```python
class Animal:
    def __init__(self, name):
        self.name = name
        print(f"Animal init: {name}")

class Mammal(Animal):
    def __init__(self, name, warm_blooded=True):
        super().__init__(name)  # Calls Animal.__init__
        self.warm_blooded = warm_blooded
        print(f"Mammal init: {name}")

class Dog(Mammal):
    def __init__(self, name, breed):
        super().__init__(name)  # Calls Mammal.__init__
        self.breed = breed
        print(f"Dog init: {name}, {breed}")

# Creating a dog follows the MRO chain
my_dog = Dog("Buddy", "Golden Retriever")
# Output:
# Animal init: Buddy
# Mammal init: Buddy  
# Dog init: Buddy, Golden Retriever
```

### 3. Advanced Sorting Techniques

#### Sorting Objects
```python
class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age
    
    def __repr__(self):
        return f"Student({self.name}, {self.grade}, {self.age})"

students = [
    Student("Alice", 85, 20),
    Student("Bob", 92, 19),
    Student("Charlie", 85, 21)
]

# Sort by grade (descending), then by age (ascending) as tiebreaker
sorted_students = sorted(students, key=lambda s: (-s.grade, s.age))
print(sorted_students)
# Output: [Student(Bob, 92, 19), Student(Alice, 85, 20), Student(Charlie, 85, 21)]
```

#### Custom Sorting Functions
```python
# Sort strings by second character
words = ["apple", "banana", "cherry", "date"]
sorted_words = sorted(words, key=lambda x: x[1])
print(sorted_words)  # ['date', 'cherry', 'apple', 'banana']

# Sort tuples by multiple criteria
points = [("A", 3, 4), ("B", 1, 2), ("C", 3, 1), ("D", 1, 5)]

# Sort by second element, then third element
sorted_points = sorted(points, key=lambda p: (p[1], p[2]))
print(sorted_points)  # [('B', 1, 2), ('D', 1, 5), ('C', 3, 1), ('A', 3, 4)]

# Sort by distance from origin
import math
sorted_by_distance = sorted(points, key=lambda p: math.sqrt(p[1]**2 + p[2]**2))
print(sorted_by_distance)
```

#### In-place vs Copy Sorting
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# sorted() returns a new list
sorted_copy = sorted(numbers)
print(f"Original: {numbers}")      # [3, 1, 4, 1, 5, 9, 2, 6]
print(f"Sorted copy: {sorted_copy}")  # [1, 1, 2, 3, 4, 5, 6, 9]

# sort() modifies the original list
numbers.sort()
print(f"After sort(): {numbers}")  # [1, 1, 2, 3, 4, 5, 6, 9]
```

### 4. Exception Handling Patterns

#### Basic Try-Except Structure
```python
def safe_divide(a, b):
    try:
        result = a / b
        print(f"Division successful: {result}")
        return result
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")
        return None
    except TypeError:
        print("Error: Invalid types for division!")
        return None
    finally:
        print("Division operation completed")

# Test cases
print(safe_divide(10, 2))    # 5.0
print(safe_divide(10, 0))    # None (ZeroDivisionError)
print(safe_divide("10", 2))  # None (TypeError)
```

#### Multiple Exception Handling
```python
def process_file(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read()
            number = int(data.strip())
            result = 100 / number
            return result
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except ValueError:
        print("Error: File content is not a valid number")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero")
    except (IOError, OSError) as e:
        print(f"Error reading file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("File processing completed")
```

#### Custom Exceptions
```python
class CustomValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message, error_code=None):
        super().__init__(message)
        self.error_code = error_code
    
    def __str__(self):
        if self.error_code:
            return f"Validation Error [{self.error_code}]: {self.args[0]}"
        return f"Validation Error: {self.args[0]}"

class UserValidator:
    @staticmethod
    def validate_age(age):
        if not isinstance(age, int):
            raise CustomValidationError("Age must be an integer", "TYPE_ERROR")
        if age < 0:
            raise CustomValidationError("Age cannot be negative", "VALUE_ERROR")
        if age > 150:
            raise CustomValidationError("Age seems unrealistic", "RANGE_ERROR")
        return True

# Usage
try:
    UserValidator.validate_age(-5)
except CustomValidationError as e:
    print(e)  # Validation Error [VALUE_ERROR]: Age cannot be negative
```

### 5. Advanced OOP Patterns

#### Property Decorators and Encapsulation
```python
class BankAccount:
    def __init__(self, initial_balance=0):
        self._balance = initial_balance  # Protected attribute
        self.__account_number = self._generate_account_number()  # Private
    
    @property
    def balance(self):
        """Getter for balance"""
        return self._balance
    
    @balance.setter
    def balance(self, amount):
        """Setter with validation"""
        if amount < 0:
            raise ValueError("Balance cannot be negative")
        self._balance = amount
    
    @property
    def account_number(self):
        """Read-only property"""
        return self.__account_number
    
    def _generate_account_number(self):
        """Protected method"""
        import random
        return f"ACC{random.randint(100000, 999999)}"
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

# Usage
account = BankAccount(1000)
print(account.balance)        # 1000
print(account.account_number) # ACC123456 (read-only)

account.deposit(500)
print(account.balance)        # 1500

try:
    account.balance = -100    # Triggers setter validation
except ValueError as e:
    print(e)  # Balance cannot be negative
```

---

## 🎤 Part 3: Interview Questions & Detailed Answers

### Basic Level Questions

#### Q1: What is the difference between `sorted()` and `sort()` in Python?

**Answer:**

**`sorted()` Function:**
- Returns a **new sorted list**
- Does **not modify** the original list
- Can be used on **any iterable** (lists, tuples, strings)
- **Function signature**: `sorted(iterable, key=None, reverse=False)`

**`sort()` Method:**
- **Modifies the original list** in-place
- Returns `None`
- Only available on **list objects**
- **Method signature**: `list.sort(key=None, reverse=False)`

**Example:**
```python
numbers = [3, 1, 4, 1, 5]

# Using sorted() - creates new list
sorted_numbers = sorted(numbers)
print(f"Original: {numbers}")        # [3, 1, 4, 1, 5]
print(f"Sorted copy: {sorted_numbers}")  # [1, 1, 3, 4, 5]

# Using sort() - modifies original
numbers.sort()
print(f"After sort(): {numbers}")    # [1, 1, 3, 4, 5]
```

**When to use which:**
- Use `sorted()` when you need to keep the original list unchanged
- Use `sort()` when you want to save memory and don't need the original order

#### Q2: Explain polymorphism in Python with examples.

**Answer:**

**Polymorphism** means "many forms" - the ability of objects of different types to be treated as instances of the same type through a common interface.

**Types of Polymorphism in Python:**

**1. Dynamic Polymorphism (Method Overriding):**
```python
class Shape:
    def area(self):
        pass
    
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Polymorphic behavior
shapes = [Rectangle(5, 3), Circle(4), Rectangle(2, 8)]

for shape in shapes:
    print(f"Area: {shape.area()}")  # Different implementations called
```

**2. Static Polymorphism (Function Overloading Simulation):**
```python
class Calculator:
    def add(self, *args):
        if len(args) == 2:
            return args[0] + args[1]
        elif len(args) == 3:
            return args[0] + args[1] + args[2]
        else:
            return sum(args)

calc = Calculator()
print(calc.add(1, 2))        # 3
print(calc.add(1, 2, 3))     # 6
print(calc.add(1, 2, 3, 4))  # 10
```

#### Q3: What is Method Resolution Order (MRO) and why is it important?

**Answer:**

**Method Resolution Order (MRO)** is the order in which Python searches for methods in a class hierarchy, especially important in multiple inheritance scenarios.

**Why MRO Matters:**
- Determines which method gets called when there are multiple definitions
- Ensures consistent and predictable behavior
- Prevents ambiguity in complex inheritance hierarchies

**MRO Algorithm (C3 Linearization):**
1. Start with the current class
2. Follow the inheritance chain left-to-right
3. Ensure each class appears only once
4. Maintain the order specified in inheritance declarations

**Example:**
```python
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):  # Multiple inheritance
    pass

# Check MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
print(d.method())  # "B" - follows MRO order
```

**Using `super()` with MRO:**
```python
class A:
    def __init__(self):
        print("A init")

class B(A):
    def __init__(self):
        print("B init")
        super().__init__()

class C(A):
    def __init__(self):
        print("C init")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D init")
        super().__init__()

D()  # Output: D init, B init, C init, A init
```

### Intermediate Level Questions

#### Q4: How do you implement custom sorting for complex objects?

**Answer:**

**Custom sorting** allows you to define how objects should be compared and ordered based on specific attributes or criteria.

**Methods for Custom Sorting:**

**1. Using `key` parameter with lambda:**
```python
class Employee:
    def __init__(self, name, salary, department):
        self.name = name
        self.salary = salary
        self.department = department
    
    def __repr__(self):
        return f"Employee({self.name}, ${self.salary}, {self.department})"

employees = [
    Employee("Alice", 75000, "Engineering"),
    Employee("Bob", 65000, "Marketing"),
    Employee("Charlie", 75000, "Engineering"),
    Employee("Diana", 80000, "Engineering")
]

# Sort by salary (descending), then by name (ascending)
sorted_employees = sorted(employees, key=lambda e: (-e.salary, e.name))
print(sorted_employees)
```

**2. Using `operator.attrgetter` for cleaner code:**
```python
from operator import attrgetter

# Sort by department, then salary
sorted_by_dept = sorted(employees, key=attrgetter('department', 'salary'))

# Sort by multiple attributes with custom logic
def custom_key(employee):
    return (employee.department, -employee.salary, employee.name)

sorted_custom = sorted(employees, key=custom_key)
```

**3. Implementing comparison methods:**
```python
from functools import total_ordering

@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    
    def __repr__(self):
        return f"Student({self.name}, {self.grade})"

students = [Student("Alice", 85), Student("Bob", 92), Student("Charlie", 78)]
sorted_students = sorted(students)  # Uses __lt__ for comparison
```

#### Q5: Explain exception handling best practices in Python.

**Answer:**

**Exception Handling Best Practices:**

**1. Be Specific with Exception Types:**
```python
# Bad - catches everything
try:
    result = int(user_input) / divisor
except:
    print("Something went wrong")

# Good - specific exception handling
try:
    result = int(user_input) / divisor
except ValueError:
    print("Invalid input: not a number")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Unexpected error: {e}")
```

**2. Use Finally for Cleanup:**
```python
def process_file(filename):
    file_handle = None
    try:
        file_handle = open(filename, 'r')
        data = file_handle.read()
        return process_data(data)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None
    finally:
        if file_handle:
            file_handle.close()  # Always cleanup
```

**3. Context Managers (Preferred):**
```python
def process_file(filename):
    try:
        with open(filename, 'r') as file:  # Automatic cleanup
            data = file.read()
            return process_data(data)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except IOError as e:
        print(f"Error reading file: {e}")
        return None
```

**4. Custom Exceptions for Domain Logic:**
```python
class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class BusinessLogicError(Exception):
    """Raised when business rules are violated"""
    pass

def validate_user_age(age):
    if not isinstance(age, int):
        raise ValidationError("Age must be an integer")
    if age < 0 or age > 150:
        raise ValidationError("Age must be between 0 and 150")
    return True

def create_user(name, age):
    try:
        validate_user_age(age)
        if age < 18:
            raise BusinessLogicError("User must be 18 or older")
        return User(name, age)
    except ValidationError as e:
        print(f"Validation failed: {e}")
    except BusinessLogicError as e:
        print(f"Business rule violation: {e}")
```

### Advanced Level Questions

#### Q6: How does Python's `super()` work in complex inheritance hierarchies?

**Answer:**

**`super()` in Python** follows the Method Resolution Order (MRO) to call methods in parent classes, which is crucial for cooperative inheritance.

**How `super()` Works:**

**1. Single Inheritance:**
```python
class Animal:
    def __init__(self, name):
        self.name = name
        print(f"Animal.__init__: {name}")

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)  # Calls Animal.__init__
        self.breed = breed
        print(f"Dog.__init__: {name}, {breed}")

dog = Dog("Buddy", "Golden Retriever")
# Output:
# Animal.__init__: Buddy
# Dog.__init__: Buddy, Golden Retriever
```

**2. Multiple Inheritance (Diamond Problem):**
```python
class A:
    def __init__(self):
        print("A.__init__")

class B(A):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A):
    def __init__(self):
        print("C.__init__")
        super().__init__()

class D(B, C):
    def __init__(self):
        print("D.__init__")
        super().__init__()

print("MRO:", D.__mro__)
# MRO: (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

D()
# Output:
# D.__init__
# B.__init__
# C.__init__
# A.__init__
```

**3. Cooperative Inheritance Pattern:**
```python
class Shape:
    def __init__(self, **kwargs):
        print("Shape.__init__")
        super().__init__(**kwargs)  # Pass remaining kwargs up the chain

class Colorable:
    def __init__(self, color="white", **kwargs):
        self.color = color
        print(f"Colorable.__init__: color={color}")
        super().__init__(**kwargs)

class Rectangle(Shape, Colorable):
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
        print(f"Rectangle.__init__: {width}x{height}")
        super().__init__(**kwargs)

# All classes cooperatively call super()
rect = Rectangle(10, 5, color="red")
# Output:
# Rectangle.__init__: 10x5
# Shape.__init__
# Colorable.__init__: color=red
```

#### Q7: Implement a decorator that handles exceptions and retries operations.

**Answer:**

**Exception Handling Decorator with Retry Logic:**

```python
import time
import functools
from typing import Callable, Type, Tuple

def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator that retries a function on specified exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        print(f"Function {func.__name__} failed after {max_retries} retries")
                        raise e
                    
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
                except Exception as e:
                    # Don't retry on unexpected exceptions
                    print(f"Unexpected exception in {func.__name__}: {e}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

# Usage examples
@retry_on_exception(max_retries=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def unreliable_network_call():
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network connection failed")
    return "Success!"

@retry_on_exception(max_retries=2, delay=1.0, exceptions=(ValueError,))
def parse_data(data):
    if not data.strip():
        raise ValueError("Empty data")
    return int(data)

# Test the decorators
try:
    result = unreliable_network_call()
    print(f"Network call result: {result}")
except ConnectionError as e:
    print(f"Final failure: {e}")

try:
    number = parse_data("  ")
except ValueError as e:
    print(f"Parse failure: {e}")
```

**Advanced Exception Context Manager:**

```python
import logging
from contextlib import contextmanager
from typing import Type, Optional

@contextmanager
def exception_handler(
    exception_type: Type[Exception] = Exception,
    default_return: any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    Context manager for handling exceptions with logging and default returns.
    """
    try:
        yield
    except exception_type as e:
        if log_errors:
            logging.error(f"Exception caught: {type(e).__name__}: {e}")
        
        if reraise:
            raise e
        
        return default_return

# Usage
def risky_operation():
    with exception_handler(ValueError, default_return="Error occurred", log_errors=True):
        result = int("not_a_number")
        return result
    return "This won't be reached if exception occurs"

print(risky_operation())  # "Error occurred"
```

---

## 🚀 Practical Tips for Interviews

### 1. **Demonstrate Understanding with Code**
Always provide working code examples:
```python
# Show polymorphism understanding
def demonstrate_polymorphism():
    animals = [Dog(), Cat(), Bird()]
    for animal in animals:
        print(animal.make_sound())  # Different implementations
```

### 2. **Know the Trade-offs**
Be prepared to discuss:
- **sorted() vs sort()**: Memory usage vs performance
- **Exception handling**: Specific vs generic exception catching
- **Inheritance vs Composition**: When to use each approach

### 3. **Common Pitfalls to Avoid**
```python
# Bad: Catching all exceptions
try:
    risky_operation()
except:  # Don't do this!
    pass

# Good: Specific exception handling
try:
    risky_operation()
except SpecificError as e:
    handle_specific_error(e)
except Exception as e:
    log_unexpected_error(e)
    raise
```

### 4. **Show Advanced Knowledge**
```python
# Demonstrate understanding of descriptors, metaclasses, etc.
class LoggedAttribute:
    def __init__(self, name):
        self.name = name
    
    def __get__(self, obj, objtype=None):
        value = obj.__dict__.get(self.name)
        print(f"Getting {self.name}: {value}")
        return value
    
    def __set__(self, obj, value):
        print(f"Setting {self.name}: {value}")
        obj.__dict__[self.name] = value
```

---

## 📚 Key Concepts from the Meeting

### 1. **Polymorphism Types:**
- Dynamic polymorphism (method overriding)
- Static polymorphism (function overloading simulation)
- Method Resolution Order (MRO)

### 2. **Sorting Techniques:**
- `sorted()` vs `sort()` methods
- Custom key functions with lambda
- Multi-criteria sorting
- Sorting objects by attributes

### 3. **Exception Handling:**
- Try-except-finally blocks
- Specific exception types
- Custom exceptions
- Exception hierarchy and inheritance

### 4. **Advanced OOP:**
- Method resolution order
- `super()` usage in inheritance
- Protected and private attributes
- Property decorators

---

## 📊 Additional Resources

### Key Python Concepts:
1. **Object-Oriented Programming**: Classes, inheritance, polymorphism
2. **Exception Handling**: Try-except patterns, custom exceptions
3. **Sorting and Searching**: Built-in functions, custom comparators
4. **Advanced Features**: Decorators, context managers, descriptors

### Best Practices:
- Use specific exception types
- Implement `__repr__` for debugging
- Follow PEP 8 style guidelines
- Use type hints for better code documentation

### Common Interview Topics:
- Difference between `is` and `==`
- Mutable vs immutable objects
- List comprehensions vs generator expressions
- Memory management and garbage collection

---

*Remember: Python interviews often focus on practical problem-solving and understanding of core concepts. Practice implementing these patterns and be ready to explain the reasoning behind your design choices!* 🎯