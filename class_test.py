import inspect
from types import MethodType

class Student(object):
    @property
    def birth(self):
        return self._birth

    @property.setter
    def birth(self, value):
        self._birth = value

    @property
    def age(self):
        return 2017 - self._birth



s = Student()
s.birth
