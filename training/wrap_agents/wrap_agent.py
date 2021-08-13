from abc import ABC, abstractmethod
from typing import Any, List, Tuple
import tensorflow as tf


class WrapAgent(ABC):
    @abstractmethod
    def get_gamma(self):
        pass
