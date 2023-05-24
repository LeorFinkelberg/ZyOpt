import abc


class Model(abc.ABC):
    @abc.abstractmethod
    def optimize(self):
        """
        Optimize the problem
        """
