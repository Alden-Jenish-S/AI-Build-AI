"""Built-in modality adapters."""

from __future__ import annotations

from core.modality_registry import ModalityRegistry


def build_default_registry() -> ModalityRegistry:
    """Return a new registry containing every production-ready adapter."""
    from .audio import AudioAdapter
    from .image import ImageAdapter
    from .multimodal import MultimodalAdapter
    from .tabular import TabularAdapter
    from .text import TextAdapter
    from .video import VideoAdapter

    registry = ModalityRegistry()
    registry.register("tabular", TabularAdapter())
    registry.register("image", ImageAdapter())
    registry.register("audio", AudioAdapter())
    registry.register("video", VideoAdapter())
    registry.register("text", TextAdapter())
    registry.register("multimodal", MultimodalAdapter())
    return registry


__all__ = ["build_default_registry"]
