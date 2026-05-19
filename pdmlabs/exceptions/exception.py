class NotFitForSourceException(Exception):
    def __init__(self, message="'predict' was called on a model that was not previously fit for this source"):
        self.message = message
        super().__init__(self.message)


class IncompatibleMethodException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ShortScenarioLengthException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)