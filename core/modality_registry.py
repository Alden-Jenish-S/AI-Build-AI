"""Registry for modality-specific task adapters."""

from __future__ import annotations

from typing import Iterable

from .contracts import normalize_modality


class ModalityRegistry:
    """Resolve adapters without introducing modality branches in the manager."""

    def __init__(self) -> None:
        self._adapters: dict[str, object] = {}

    def register(
        self,
        modality: str,
        adapter: object,
        *,
        replace: bool = False,
    ) -> None:
        normalized = normalize_modality(modality)
        if normalized in self._adapters and not replace:
            raise ValueError(
                f"an adapter is already registered for {normalized!r}"
            )
        adapter_name = normalize_modality(getattr(adapter, "name", normalized))
        if adapter_name != normalized:
            raise ValueError(
                f"adapter name {adapter_name!r} does not match "
                f"registration {normalized!r}"
            )
        self._adapters[normalized] = adapter

    def get(self, modality: str) -> object:
        normalized = normalize_modality(modality)
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise LookupError(
                f"no adapter registered for modality {normalized!r}; "
                f"available={list(self.names())}"
            ) from exc

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._adapters))

    def __contains__(self, modality: object) -> bool:
        try:
            normalized = normalize_modality(modality)
        except ValueError:
            return False
        return normalized in self._adapters
