import numpy as np


class BinaryLinearSVM():
    def fit(self, x: np.ndarray, y: np.ndarray):

        self._targets = np.unique(y)
        assert len(self._targets) == 2

        y_ = np.copy(y)
        y_[y == self._targets[0]] = -1
        y_[y == self._targets[1]] = 1

        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        max_feature_value = np.max(x)

        step = max_feature_value * 0.05
        b_range_multiple = 10
        b_multiple = 2

        latest_optimum = max_feature_value * 10

        best_m = 100000
        best_w_t = np.array([latest_optimum, latest_optimum])
        best_b = 0

        w = np.array([latest_optimum, latest_optimum])

        optimized = False
        while not optimized:
            for b in np.arange(-1 * max_feature_value * b_range_multiple,
                               max_feature_value * b_range_multiple,
                               step * b_multiple):
                for transformation in transforms:
                    w_t = w * transformation

                    m = np.average((1 - (y_ * (np.dot(x, w_t) + b))).clip(0).astype(float))
                    if m < best_m:
                        best_m = m
                        best_w_t = w_t
                        best_b = b

            if w[0] < 0:
                optimized = True
            else:
                w = w - step * max_feature_value

        self._w = best_w_t
        self._b = best_b

    def predict(self, x):
        proba = np.dot(np.array(x), self._w) + self._b
        return [self._targets[0] if sgn < 0 else self._targets[1] for sgn in proba]

    def predict_proba(self, x):
        proba = np.dot(np.array(x), self._w) + self._b
        proba = (np.exp(proba) - np.exp(-proba)) / (np.exp(proba) + np.exp(-proba))
        proba /= 2

        return np.transpose([0.5 - proba, 0.5 + proba])