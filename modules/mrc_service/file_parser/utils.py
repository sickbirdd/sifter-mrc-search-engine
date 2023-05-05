from enum import Enum

def singleton(cls):
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return getinstance


class StrEnum(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name