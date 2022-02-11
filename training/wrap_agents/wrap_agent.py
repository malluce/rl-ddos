from abc import ABC, abstractmethod


class WrapAgent(ABC):
    @abstractmethod
    def get_gamma(self):
        pass
