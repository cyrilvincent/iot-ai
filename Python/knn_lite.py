class KNNLite:

    def __init__(self, k):
        self.k = k
        self.top_distances: list[float] = []
        self.top_indexes: list[float] = []
        self.top_labels: list[float] = []

    def load_csv(self, path: str) -> list[list[float]]:
        res = []
        with open(path) as f:
            for row in f:
                row = row.strip()
                if row:
                    values = row.split(",")
                    res.append([float(v) for v in values])
        return res

    def load_csv_1d(self, path: str) -> list[float]:
        res = []
        with open(path) as f:
            for row in f:
                row = row.strip()
                if row:
                    res.append(float(row))
        return res

    def distance2(self, v1: list[float], v2: list[float]):
        dist = 0
        for value1, value2 in zip(v1, v2):
            dist += (value1 - value2) ** 2
        return dist

    def nearing(self, distance: float, index: int, y: float):
        if not self.top_distances:
            self.top_distances = [distance]
            self.top_indexes = [index]
            self.top_labels = [y]
            return
        i = 0
        for d in self.top_distances:
            if distance < d:
                self.top_distances.insert(i, distance)
                self.top_indexes.insert(i, index)
                self.top_labels.insert(i, y)
                break
            i += 1
        if len(self.top_distances) > self.k:
            self.top_distances = self.top_distances[:self.k]
            self.top_indexes = self.top_indexes[:self.k]
            self.top_labels = self.top_labels[:self.k]

    def scale(self, x: list[list[float]], means: list[float], stds: list[float]):
        result = []
        for row in x:
            norm = [(v - mean) / std for v, mean, std in zip(row, means, stds)]
            result.append(norm)
        return result

    def train_predict(self, xtrain: list[list[float]], ytrain: list[float], x: list[float], verbose=False):
        self.top_distances = [2 ** 16 - 1] * self.k
        self.top_indexes = [-1] * self.k
        self.top_labels = [-1] * self.k
        nb = len(ytrain)
        i = 0
        for row, label in zip(xtrain, ytrain):
            if verbose:
                print(f"Train {i+1}/{nb}")
            distance = self.distance2(row, x)
            self.nearing(distance, i, label)
            i += 1
        return self.top_labels


if __name__ == '__main__':
    model = KNNLite(3)
    xtrain = model.load_csv("data/heart/x_train.csv")
    ytrain = model.load_csv_1d("data/heart/y_train.csv")
    means = model.load_csv_1d("data/heart/scaler_mean.csv")
    stds = model.load_csv_1d("data/heart/scaler_scale.csv")
    xnorm = model.scale(xtrain, means, stds)
    x = [[28, 1, 2, 130, 132, 0, 2, 185, 0, 0]]
    x = model.scale(x, means, stds)
    result = model.train_predict(xnorm, ytrain, x[0], False)
    print(result)
    x = [[65, 1, 4, 130, 275, 0, 1, 115, 1, 1]]
    x = model.scale(x, means, stds)
    result = model.train_predict(xnorm, ytrain, x[0], False)
    print(result)




