class PCALite:

    def __init__(self, kernel: int, stride: int, l1_only=False):
        self.kernel = kernel
        self.stride = stride
        self.l1_only = l1_only

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
            if not self.l1_only:
                result.append(self.l2(window))
            i += self.stride
        return result

    def reduce2(self, x: list[list[float]]) -> list[list[float]]:
        return [self.reduce1(row) for row in x]


if __name__ == '__main__':
    x = list(range(30))
    pca = PCALite(10, 5)
    r1 = pca.reduce1(x)
    print(r1)
    x = [x]
    r2 = pca.reduce2(x)
    print(r2)
