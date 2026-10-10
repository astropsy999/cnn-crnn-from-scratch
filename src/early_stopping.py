"""Ранняя остановка: учимся, пока метрика улучшается; не улучшается patience эпох подряд — стоп."""


class EarlyStopping:
    """Следит за метрикой «больше — лучше». При равенстве метрик лучшим считается кроп с меньшим loss."""

    def __init__(self, patience: int, tol: float = 1e-9) -> None:
        if patience < 1:
            raise ValueError("patience должен быть >= 1")
        self.patience = patience
        self.tol = tol
        self.best_score = None
        self.best_loss = None
        self.bad_epochs = 0

    def step(self, score: float, loss: float) -> bool:
        """Сообщает результат эпохи. Возвращает True, если это лучшая эпоха на данный момент."""
        improved = (
            self.best_score is None
            or score > self.best_score + self.tol
            or (abs(score - self.best_score) <= self.tol and loss < self.best_loss)
        )
        if improved:
            self.best_score, self.best_loss = score, loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
