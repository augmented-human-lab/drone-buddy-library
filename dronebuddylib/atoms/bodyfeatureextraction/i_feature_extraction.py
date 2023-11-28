from abc import abstractmethod

from dronebuddylib.models.i_dbl_function import IDBLFunction


class IFeatureExtraction(IDBLFunction):
    """
    The IFeatureExtraction interface declares the method for extracting features from an image.
    """
    def __init__(self):
        pass

    @abstractmethod
    def get_feature(self, image) -> list:
        """
        Extract features from an image.

        Args:
            image: The image from which features should be extracted.

        Returns:
            A list containing the extracted features.
        """
        pass
