class ColumnsNotMatchException(Exception):
    def __init__(self, pattern, message='Columns doesn\'t match the pattern.') -> None:
        super().__init__(f'{message} {pattern}')

class SegmentationNotSupportedException(Exception):
    def __init__(self, message='Number of supported group is between 1 and 3.') -> None:
        super().__init__(message)

class WrongProportionsException(Exception):
    def __init__(self, message='Proportions doesn\'t cover all the items/users.') -> None:
        super().__init__(message)

class FlagNotValidException(Exception):
    def __init__(self, message='Invalid flag.') -> None:
        super().__init__(message)
