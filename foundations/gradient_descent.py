class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        while iterations>0:
            fx=2*init
            init=init-fx*learning_rate
            iterations=iterations-1
        return round(init,5)
        pass
