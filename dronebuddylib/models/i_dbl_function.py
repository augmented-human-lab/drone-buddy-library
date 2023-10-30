from abc import ABC, abstractmethod


class IDBLFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_required_params(self) -> list:
        pass
    @abstractmethod
    def get_optional_params(self) -> list:
        pass

    @abstractmethod
    def get_class_name(self) -> str:
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        pass
