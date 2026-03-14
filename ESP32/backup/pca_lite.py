class PCALite:

    def __init__(self, kernel: int, stride: int):
        self.kernel = kernel
        self.stride = stride

    def l1(self, x: list[float]) -> float:
        return sum(x) / len(x)

    def l2(self, x: list[float]) -> float:
        return sum([v ** 2 for v in x]) ** 0.5

    def reduce1(self, x: list[float]) -> list[float]:
        result = []
        x = list(x)
        i = 0
        while i < len(x):
            j = (i + self.stride) % len(x)
            if i < j:
                window = x[i:j]
            else:
                window = x[i:] + x[:j]
            result.append(self.l1(window))
            result.append(self.l2(window))
            i += self.stride
        return result

    def reduce2(self, x: list[list[float]]) -> list[list[float]]:
        return [self.reduce1(row) for row in x]


