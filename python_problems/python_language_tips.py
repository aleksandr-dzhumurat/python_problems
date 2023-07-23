# decorator

def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

def number_generator():
    """generator example"""
    yield 1
    yield 2
    yield 3

def test_generator():
    # Create a generator object
    gen = number_generator()

    # Iterate over the generator using a for loop
    for num in gen:
        print(num)
