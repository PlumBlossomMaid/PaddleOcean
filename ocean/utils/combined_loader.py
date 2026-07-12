"""Combined data loader for multiple dataloaders."""

from typing import Any, Iterator

from ocean.utils.exceptions import MisconfigurationException


class CombinedLoader(Iterator):
    """Wrapper for iterating over multiple dataloaders sequentially or in parallel.

    Args:
        loaders: Single dataloader or list of dataloaders.
        mode: 'sequential' or 'min_size' or 'max_size'.
    """

    def __init__(self, loaders: Any, mode: str = "sequential") -> None:
        if mode != "sequential":
            raise MisconfigurationException(
                f"CombinedLoader mode {mode!r} is not implemented. Only "
                "'sequential' is supported; 'min_size'/'max_size' require a "
                "value-aware merge that isn't wired up, so reject up front."
            )
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        self.loaders = loaders
        self.mode = mode
        self._iterators: list = []
        self._current_idx = 0

    @property
    def flattened(self) -> list:
        """Flat ``[loader, ...]`` — the statefulness probe point.

        ``_state_dicts``/``_load_state_dicts`` iterate this to collect and restore
        each loader's state. Statefulness is decided duck-typed on the existence of
        ``state_dict``/``load_state_dict``: a loader that wants to survive a
        checkpoint implements them, and that is what gets persisted.
        """
        return [ld for ld in self.loaders if ld is not None]

    def _state_dicts(self) -> list:
        """State dicts of the stateful loaders (skip non-stateful silently).

        Collect ``loader.state_dict()`` only from loaders that expose it. Loaders
        without that capability are skipped without error, so a plain dataloader
        that has nothing to persist simply contributes nothing to the checkpoint.
        """
        return [
            ld.state_dict()
            for ld in self.flattened
            if hasattr(ld, "state_dict") and callable(getattr(ld, "state_dict", None))
        ]

    def _load_state_dicts(self, states: list) -> None:
        """Restore stored state dicts to the stateful loaders.

        No-op when there is nothing to restore. Otherwise the number of loaders
        that can receive state must match the number of stored states; a mismatch
        means the dataloaders changed between save and resume and raises
        ``RuntimeError`` rather than silently restoring into the wrong loaders.
        """
        if not states:
            return
        stateful_loaders = [
            ld
            for ld in self.flattened
            if hasattr(ld, "load_state_dict") and callable(getattr(ld, "load_state_dict", None))
        ]
        if len(stateful_loaders) != len(states):
            raise RuntimeError(
                f"The CombinedLoader has {len(stateful_loaders)} stateful loaders, but found {len(states)} states"
                " in the checkpoint. Please make sure you use the same dataloaders that were used when saving"
                " the checkpoint."
            )
        for ld, state in zip(stateful_loaders, states):
            ld.load_state_dict(state)

    def __iter__(self) -> "CombinedLoader":
        self._iterators = [iter(loader) for loader in self.loaders if loader is not None]
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
