class Animal(object):
    def run(self):
        print 'Animal is running...'

class Dog(Animal):
    pass

class Cat(Animal):
    pass

class Tortoise(Animal):
    def run(self):
        print 'Tortoise is running slowly...'

def run_twice(animal):
    animal.run()
    animal.run()


dog = Dog()
dog.run()
print isinstance(dog, Animal)
print isinstance(dog, Dog)
run_twice(Cat())
Tortoise().run()
run_twice(Tortoise())
