class LinearRegression:
    def __init__(self, learning_rate=0.000005, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        self.loss_history = []

    def fit(self, x_train, y_train, z_train):
        for _ in range(self.epoch):
            predictions = self.predict(x_train, y_train)
            loss = self.mean_square_error(predictions, z_train)
            self.loss_history.append(loss)

            d_m1, d_m2, d_b = self.gradient(x_train, y_train, predictions, z_train)
            self.m1 -= self.learning_rate * d_m1
            self.m2 -= self.learning_rate * d_m2
            self.b -= self.learning_rate * d_b

    def predict(self, x, y):
        return self.m1 * x + self.m2 * y + self.b

    def mean_square_error(self, predictions, targets):
        return sum((predictions - targets) ** 2) / len(targets)

    def gradient(self, x, y, predictions, targets):
        d_m1 = -2 * sum((targets - predictions) * x) / len(targets)
        d_m2 = -2 * sum((targets - predictions) * y) / len(targets)
        d_b = -2 * sum(targets - predictions) / len(targets)
        return d_m1, d_m2, d_b
