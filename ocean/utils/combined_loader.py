"""Combined data loader for multiple dataloaders."""

from typing import Any, Iterator, Optional


class CombinedLoader(Iterator):
    """Wrapper for iterating over multiple dataloaders sequentially or in parallel.

    Args:
        loaders: Single dataloader or list of dataloaders.
        mode: 'sequential' or 'min_size' or 'max_size'.
    """

    def __init__(self, loaders: Any, mode: str = "sequential") -> None:
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        self.loaders = loaders
        self.mode = mode
        self._iterators: list = []
        self._current_idx = 0

    def __iter__(self) -> "CombinedLoader":
        self._iterators = [iter(l) for l in self.loaders if l is not None]
        self._current_idx = 0
        return self

    def __next__(self) -> Any:
        if self.mode == "sequential":
            return self._next_sequential()
        return self._next_batch()

    def _next_sequential(self) -> Any:
        while self._current_idx < len(self._iterators):
            try:
                return next(self._iterators[self._current_idx])
            except StopIteration:
                self._current_idx += 1
        raise StopIteration

    def _next_batch(self) -> Any:
        raise NotImplementedError("Only sequential mode is implemented")
