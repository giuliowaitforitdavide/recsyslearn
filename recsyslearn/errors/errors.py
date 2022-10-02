
class ColumnsNotExistException(Exception):
    def __init__(self, columns, message="Dataframe does not contain columns.") -> None:
        super().__init__(f"{message} {columns}")


class RecListTooShortException(Exception):
    def __init__(self, k, message="Recommendation list too short.") -> None:
        super().__init__(f"{message} k={k}")


class SegmentationNotSupportedException(Exception):
    def __init__(self, message="Number of supported group is between 1 and 3.") -> None:
        super().__init__(message)


class WrongProportionsException(Exception):
    def __init__(self, message="Proportions doesn't cover all the items/users.") -> None:
        super().__init__(message)


class FlagNotValidException(Exception):
    def __init__(self, message="Invalid flag.") -> None:
        super().__init__(message)


class InvalidValueException(Exception):
    def __init__(self, fill_na) -> None:
        message = f"Feature contains {fill_na} as value. Please select another fill_na value."
        super().__init__(message)
