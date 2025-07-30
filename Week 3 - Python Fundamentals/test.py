class Test:
    num_objects = 0
    count_static = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def increment_objects(self):
        self.num_objects += 1

    @classmethod
    def get_objects(self):
        return self.num_objects
    
    @staticmethod
    def increment_static():
        Test.count_static += 1

    @staticmethod
    def get_static():
        return Test.count_static
    
    
test = Test("John", 25)
another_test = Test("Jane", 30)

print(Test.count_static)
print(Test.num_objects)
test.increment_objects()
test.increment_static()
print(Test.num_objects)

another_test.increment_objects()
another_test.increment_static()

print(Test.count_static)
print(test.num_objects)
print(test.get_objects())
print(another_test.get_objects())
print(another_test.get_static())

