# Python retro

## 1. Basic Python Concepts
- **Q1**: What are Python's key features? Why is Python called an interpreted language?
  - **A1**: Python is dynamically typed, interpreted, and supports object-oriented and functional programming. It's called interpreted because Python code is executed line-by-line by an interpreter.

- **Q2**: Explain the difference between `list`, `tuple`, and `set` in Python.
  - **A2**: `List`: Ordered, mutable. `Tuple`: Ordered, immutable. `Set`: Unordered, mutable, unique elements.

- **Q3**: How does Python's memory management work? What is garbage collection?
  - **A3**: Python uses a private heap to store objects. The `gc` module helps with garbage collection using reference counting and cyclic garbage collector.

## 2. Data Structures
- **Q1**: How do you implement a stack and queue in Python?
  - **A1**: Implement stack using a list (`append`/`pop`), and queue using `collections.deque`.

- **Q2**: How does Python handle dictionaries? Explain how dictionaries are optimized for lookups.
  - **A2**: Dictionaries use a hash table for fast key-value lookups (O(1) on average).

- **Q3**: What are list comprehensions and when would you use them?
  - **A3**: List comprehensions provide a concise way to create lists using loops and conditions.

## 3. Functions and OOP
- **Q1**: Explain the difference between *args and **kwargs in Python.
  - **A1**: `*args` collects positional arguments as a tuple, `**kwargs` collects keyword arguments as a dictionary.

- **Q2**: What is the difference between a shallow copy and a deep copy? How can you create them?
  - **A2**: Shallow copy copies only references to objects, deep copy duplicates everything recursively.

- **Q3**: How does Python's method resolution order (MRO) work in multiple inheritance?
  - **A3**: MRO determines the order in which base classes are searched when invoking methods, following the C3 Linearization algorithm.

## 4. Error Handling and Debugging
- **Q1**: How does Python's `try-except` block work? Can you catch multiple exceptions at once?
  - **A1**: `try-except` blocks handle exceptions, and multiple exceptions can be caught using tuples (`except (TypeError, ValueError)`).

- **Q2**: How do you debug a Python program?
  - **A2**: Use debugging tools like `pdb`, IDE debuggers, or print/logging for tracing errors.

- **Q3**: What is the use of the `finally` block in exception handling?
  - **A3**: `finally` ensures code execution regardless of whether an exception occurred.

## 5. Libraries and Modules
- **Q1**: What is the difference between a module and a package in Python?
  - **A1**: A module is a single file of Python code, while a package is a collection of modules organized in directories.

- **Q2**: How would you use `virtualenv` or `venv` to manage dependencies?
  - **A2**: `virtualenv` and `venv` are used to create isolated Python environments to manage dependencies.

- **Q3**: Explain how you would use `requests` or `urllib` to make HTTP requests.
  - **A3**: `requests` provides a simpler interface for making HTTP requests, while `urllib` is more low-level.

## 6. File I/O and Serialization
- **Q1**: How do you read and write files in Python? What are the different modes for file I/O?
  - **A1**: Open files with `open()`, using modes like `'r'` (read), `'w'` (write), `'a'` (append), `'b'` (binary).

- **Q2**: What is the difference between JSON and Pickle in Python? How would you use them to serialize data?
  - **A2**: JSON is human-readable, text-based, and language-independent; `Pickle` serializes Python objects (binary format, not human-readable).

## 7. Concurrency and Parallelism
- **Q1**: What is the `GIL` (Global Interpreter Lock), and how does it affect Python's multithreading?
  - **A1**: The Global Interpreter Lock (GIL) ensures only one thread executes Python bytecode at a time, limiting true parallelism in threads.

- **Q2**: Explain the difference between `threading` and `multiprocessing` in Python.
  - **A2**: `threading` is for I/O-bound tasks, `multiprocessing` is for CPU-bound tasks (uses separate processes).

- **Q3**: What is an `asyncio` task, and how does async/await work in Python?
  - **A3**: `asyncio` allows asynchronous programming with tasks that yield control using `async` and `await`.

## 8. Algorithms and Problem-Solving
- **Q1**: Write a Python program to reverse a string recursively.
  ```python
  def reverse_string(s):
      if len(s) == 0:
          return s
      return reverse_string(s[1:]) + s[0]

 ## Underscore Concpets ##




 ## Single Leading Underscore (_method_name)
Convention for "Private" Methods:
A single leading underscore signifies that a method is intended for internal use within the class or module where it's defined. It's a convention, not a strict enforcement mechanism, meaning you can still technically call _method_name from outside the class.
Signaling Internal Implementation Details:
This convention communicates to other developers that the method is part of the class's internal implementation and should not be directly relied upon by external code. 
Example:
Python
```
class MyClass:
    def __init__(self, value):
        self.value = value

    def public_method(self):
        """This method is intended for external use."""
        self._internal_helper()
        print(f"Public method called. Value: {self.value}")

    def _internal_helper(self):
        """This method is for internal use only."""
        print("Internal helper method called.")
        self.value += 1


```
## Double Leading Underscore (__method_name)
Name Mangling:
A double leading underscore triggers a mechanism called "name mangling." Python automatically renames the method by prepending the class name (e.g., _ClassName__method_name). This helps prevent name clashes in subclasses, especially when multiple inheritance is involved.
Stronger "Privacy" Indication:
While still not truly private, name mangling makes it harder to accidentally access or override these methods from subclasses or external code, as their names are changed.
Example:
Python
```
class BaseClass:
    def __init__(self):
        self.__secret_method()

    def __secret_method(self):
        print("BaseClass's secret method.")

class DerivedClass(BaseClass):
    def __secret_method(self): # This creates a *new* method, not overriding the base class's
        print("DerivedClass's secret method.")
```
obj_base = BaseClass()
# obj_base.__secret_method() # This would raise an AttributeError

obj_derived = DerivedClass()
# obj_derived.__secret_method() # This would raise an AttributeError


### Double Leading and Trailing Underscores (__method_name__) => Operaytor overloading exp + or any thing
Special/Magic Methods (Dunder Methods):
Methods with double leading and trailing underscores are reserved for special or "magic" methods in Python, often referred to as "dunder methods" (double underscore methods). These methods define how objects of a class interact with built-in operations and functions (e.g., __init__, __str__, __add__).
## Predefined Behavior:
You should generally avoid creating your own methods using this naming convention unless you are implementing or overriding a specific dunder method to customize behavior.
Magic methods in Python, also known as "dunder methods" (due to their double underscores), are special methods that allow you to define how your custom objects interact with built-in Python functions and operators. They enable you to customize the behavior of your classes and make them behave like built-in types.
Here's a breakdown of their key aspects:
## Syntax:
Magic methods are identified by their names, which always start and end with double underscores, such as __init__, __str__, __add__, etc.

## Purpose:
They allow for "operator overloading," meaning you can define how standard operators (like +, -, *, ==) and 
built-in functions (like len(), str(), print()) behave when applied to instances of your custom classes.

```
class BaseClass:
    def __init__(self):
        self.__secret_method()

    def __secret_method(self):
        print("BaseClass's secret method.")

class DerivedClass(BaseClass):
    def __secret_method(self): # This creates a *new* method, not overriding the base class's
        print("DerivedClass's secret method.")



obj_base = BaseClass()
obj_derived = DerivedClass()

obj_base._BaseClass__secret_method()
obj_derived._BaseClass__secret_method()
obj_derived._DerivedClass__secret_method()



output
BaseClass's secret method.
BaseClass's secret method.
BaseClass's secret method.
BaseClass's secret method.
DerivedClass's secret method.
```


------------------------------------------------------------------------------------------------------------------------------


## explain
```
explain full concept class D:
    def __init__(self):
        self.public = "public"
        self._protected ="protected"
        self.__private__="private"
        self.__private=" mre private"

M = D()
print(M.public)
print(M._protected)
print(M.__private__)
print(M.__private)
print(M._D__private)

## Answer
This code snippet demonstrates Python's approach to access modifiers and encapsulation. Unlike languages such as Java or C++, Python does not have strict keywords like private or protected. Instead, it uses naming conventions and a mechanism called name mangling to simulate these behaviors.

Here is the detailed breakdown of the concepts and what happens when you run each print statement.

## 1. Public Attributes
Code: self.public = "public"
Access: print(M.public)

## Concept: In Python, any attribute that does not start with an underscore is considered public. It can be accessed and modified freely from anywhere in the code, both inside and outside the class.
Result: This works perfectly. Output: public.
## 2. Protected Attributes (Convention)
Code: self._protected = "protected"
Access: print(M._protected)

## Concept: A single underscore prefix is a convention used to tell other programmers: "This is an internal implementation detail. Please do not access or modify this from outside the class unless you know what you are doing."
## Reality: Python does not enforce this restriction. You can technically access it just like a public variable.
## Result: This works fine. Output: protected. (However, a linter or IDE might warn you that you are accessing a protected member).
## 3. The "Magic" Attribute (The Trap)
Code: self.__private__ = "private"
Access: print(M.__private__)

##  Concept: This is a common source of confusion. Double underscores at both the beginning and the end of a name are used for Python's built-in "magic" or "dunder" methods (like __init__, __str__).
##  Crucial Rule: Name mangling (the "private" mechanism) only happens if the name ends without underscores. Because this variable ends with double underscores, Python treats it as a normal public variable.
Result: This works. Output: private.
4. Private Attributes (Name Mangling)
Code: self.__private = " mre private"
Access: print(M.__private)

##  Concept: When an attribute starts with two underscores and does not end with two underscores, Python triggers Name Mangling.
##  How it works: Python changes the internal name of the variable to _ClassName__VariableName. In this case, inside the object M, the variable is actually stored as _D__private.
The Result: When you try to print M.__private, Python looks for an attribute literally named __private. Because the variable was renamed to _D__private, this attribute does not exist.
Result: This raises an AttributeError. The script will crash here.
##  5. Accessing the Mangled Name
Code: print(M._D__private)

##  Concept: Because we know Python renamed __private to _D__private, we can access it directly using its mangled name. This demonstrates that Python privacy is not strict security; it is a mechanism to prevent accidental name collisions in inheritance.
Result: This works. Output: mre private.
Summary of Execution
If you run the code block provided, here is exactly what will happen line by line:

print(M.public) $\rightarrow$ Works. Prints public.
print(M._protected) $\rightarrow$ Works. Prints protected.
print(M.__private__) $\rightarrow$ Works. Prints private. (Treated as a regular variable).
print(M.__private) $\rightarrow$ CRASHES. Raises AttributeError: 'D' object has no attribute '__private'.
Because the script crashes at line 4, the final line (print(M._D__private)) will never execute unless you remove or comment out the line causing the crash.

##  Corrected Concept
If you want to create a true "private" variable in Python (one that triggers name mangling), you should avoid the trailing underscores:


# Correct convention for private variables
self.__private_data = "secret"  # Becomes _D__private_data internally

--------------------------------------------------------------------------------------------------------------------------
```


## MORE PROVATE PUBLIC Discussion ##

```
Got it 👍 Let’s break this down step by step.

---

### 🔹 1. The Code You Shared

```python
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
```

#### What’s Happening Here?

* **`class NeuralNetwork(nn.Module):`**

  * This defines a new class `NeuralNetwork` that **inherits** from `nn.Module` (PyTorch’s base class for all neural networks).
  * By inheriting, it gets all the functionality PyTorch provides for building, saving, and running models.

* **`def __init__(self):`**

  * The constructor method of the class.
  * `__init__` is always called when you create an object:

    ```python
    model = NeuralNetwork()
    ```

    → Python runs `model.__init__()` automatically.

* **`super().__init__()`**

  * Calls the parent class (`nn.Module`)’s `__init__` method.
  * This sets up important PyTorch internals (like tracking layers/parameters).
  * If you skip this, your model won’t behave correctly because PyTorch won’t know how to manage its layers.

---

### 🔹 2. Public, Private, and Protected in Python Classes

Python doesn’t have **true** access modifiers like Java or C++. Instead, it uses **naming conventions**:

| Modifier                   | Example       | Meaning                                                                                                    |
| -------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| **Public**                 | `self.name`   | Default. Can be accessed from anywhere.                                                                    |
| **Protected** (convention) | `self._name`  | A single underscore means “internal use” but still accessible outside.                                     |
| **Private** (name-mangled) | `self.__name` | Double underscore makes it harder to access directly (Python renames it internally to `_ClassName__name`). |

---

#### Inside `__init__` you can declare:

```python
class Demo:
    def __init__(self):
        self.public_var = "I am public"
        self._protected_var = "I am protected"
        self.__private_var = "I am private"
```

* **Access examples:**

  ```python
  obj = Demo()
  print(obj.public_var)      # ✅ works
  print(obj._protected_var)  # ⚠ works but should be treated as "internal use"
  print(obj.__private_var)   # ❌ AttributeError
  print(obj._Demo__private_var)  # ✅ works (name mangling)
  ```

---

### 🔹 3. Special (`__dunder__`) Methods

* **`__init__`** → constructor (called when you create an object).
* **`__call__`** → allows the object to be called like a function.
* **`__str__`, `__repr__`** → define string representations.
* **`__len__`, `__getitem__`** → make the class behave like a container.

These are all **special methods** (a.k.a. “dunder” methods, short for *double underscore*).

---

### 🔹 4. How It All Fits in Neural Networks

A more complete PyTorch example:

```python
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # calls nn.Module.__init__()
        
        # Public attributes (layers)
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, output_size)  

        # Protected attribute (convention)
        self._activation = nn.ReLU()

        # Private attribute (name mangling)
        self.__learning_rate = 0.01

    def forward(self, x):  # defines forward pass
        x = self.fc1(x)
        x = self._activation(x)
        return self.fc2(x)
```

Usage:

```python
model = NeuralNetwork(10, 20, 2)

print(model.fc1)              # ✅ public
print(model._activation)      # ⚠ accessible but "internal"
print(model._NeuralNetwork__learning_rate)  # ⚠ hacky way to access private
```

---

✅ **Summary:**

* `__init__` sets up class variables.
* `super().__init__()` is required when subclassing `nn.Module`.
* Public/Protected/Private in Python are **conventions**, not enforced rules.
* Dunder methods (`__init__`, `__call__`, etc.) give classes special behavior.

---

Do you want me to also make a **table of common dunder methods** (like `__call__`, `__getitem__`, etc.) with PyTorch-specific examples?

```
