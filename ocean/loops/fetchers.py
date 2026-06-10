"""DataFetcher implementations for fetching batches in loops."""

from typing import Any, Iterator, Optional


class _DataFetcher(Iterator):
    """Basic DataFetcher that iterates over a CombinedLoader."""

    def __init__(self) -> None:
        self.done = False
        self._iter: Optional[Iterator] = None

    def setup(self, iterable: Any) -> None:
        if iterable is not None:
            self._iter = iter(iterable)
        self.done = False

    def __next__(self) -> Any:
        if self._iter is None:
            raise StopIteration
        try:
            return next(self._iter)
        except StopIteration:
            self.done = True
            raise

    def __iter__(self) -> "_DataFetcher":
        return self

    def teardown(self) -> None:
        self._iter = None
        self.done = False


class _PrefetchDataFetcher(_DataFetcher):
    """DataFetcher that prefetches to determine is_last_batch."""

    def __init__(self) -> None:
        super().__init__()
        self._prefetch_batch: Optional[Any] = None
        self._has_prefetched = False

    def setup(self, iterable: Any) -> None:
        super().setup(iterable)
        self._prefetch_batch = None
        self._has_prefetched = False

    def __next__(self) -> Any:
        if not self._has_prefetched:
            # First call: prefetch one batch
            try:
                self._prefetch_batch = super().__next__()
            except StopIteration:
                self.done = True
                raise
            self._has_prefetched = True

        # Return the prefetched batch and prefetch the next one
        batch = self._prefetch_batch
        try:
            self._prefetch_batch = super().__next__()
        except StopIteration:
            self._prefetch_batch = None
            self.done = True

        return batch
