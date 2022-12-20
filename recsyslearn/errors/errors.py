
class ColumnsNotExistException(Exception):

    """Exception raised when columns are not found in the DataFrame."""

    def __init__(self, columns: list, message: str = "Dataframe does not contain columns.") -> None:
        super().__init__(f"{message} {columns}")


class RecListTooShortException(Exception):

    """Exception raised when the values for the recommendation list length is not compatbile with the dataset."""

    def __init__(self, ats: int, message: str = "Values for the recommendation list length is not compatbile with the dataset.") -> None:
        super().__init__(f"{message} ats={ats}")


class SegmentationNotSupportedException(Exception):

    """Exception raised when there are too many group to be segmented."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class WrongProportionsException(Exception):

    """Exception raised when proportions of the segmentation does not cover all the items or users."""

    def __init__(self, message: str = "Proportions doesn't cover all the items/users.") -> None:
        super().__init__(message)


class InvalidValueException(Exception):

    """Exception raised when the value of fill_na of the segmentation value is invalid."""

    def __init__(self, fill_na) -> None:
        message = f"Feature contains {fill_na} as value. Please select another fill_na value."
        super().__init__(message)
