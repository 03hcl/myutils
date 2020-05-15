from abc import ABC, abstractmethod


class GraphBase(ABC):
    
    def __init__(self, *args, **kwargs):
        super(GraphBase, self).__init__()
    
    @abstractmethod
    def clear(self) -> None:
        pass
