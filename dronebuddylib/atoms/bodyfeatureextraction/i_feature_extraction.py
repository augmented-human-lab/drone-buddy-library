from abc import abstractmethod

from dronebuddylib.models.i_dbl_function import IDBLFunction


class IFeatureExtraction(IDBLFunction):
    def __init__(self):
        pass

    @abstractmethod
    def get_feature(self, image) -> list:
        pass
