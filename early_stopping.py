"""Early stopping tracks meaningful SRCC gains; checkpoint selection stays exact."""
import math


class EarlyStopping:
    def __init__(self, patience=0, min_delta=0.0):
        if patience < 0 or min_delta < 0 or not math.isfinite(min_delta):
            raise ValueError('patience and min_delta must be nonnegative and finite')
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float('inf')
        self.bad_epochs = 0

    def step(self, score):
        if math.isfinite(score) and score > self.best + self.min_delta:
            self.best = score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.patience > 0 and self.bad_epochs >= self.patience
