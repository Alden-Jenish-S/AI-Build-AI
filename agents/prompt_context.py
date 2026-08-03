"""Modality-specific correctness constraints for generated implementations."""

from __future__ import annotations

from core.contracts import TaskSpec
from evaluation.fidelity import get_fidelity_profile


def modality_prompt_context(task: TaskSpec, fidelity: str) -> str:
    """Return concise adapter-owned generation constraints."""
    profile = get_fidelity_profile(task.modality, fidelity)
    shared = (
        f"Task modality={task.modality}; components="
        f"{list(task.component_modalities)}; problem={task.problem_type}; "
        f"output={task.output.type}. Inputs are harness-indexed and sample IDs "
        "must remain aligned. Fit all learned preprocessing on fold-training "
        "samples only. Validation/inference preprocessing must be deterministic."
    )
    if task.modality == "image":
        detail = (
            f"Decode paths lazily. Bound images to {profile.spatial_size}; "
            "random crop/color/geometric augmentation is training-only. "
            "Normalize validation/test with training-defined constants."
        )
    elif task.modality == "audio":
        detail = (
            f"Decode waveforms lazily, resample inside the fold pipeline to "
            f"{profile.audio_sample_rate} Hz, and cap clips at "
            f"{profile.max_audio_seconds} seconds at this fidelity. Speaker or "
            "session groups may not cross folds. Audio augmentation is "
            "training-only."
        )
    elif task.modality == "video":
        detail = (
            f"Use deterministic validation clips with at most "
            f"{profile.video_frames} frames, {profile.video_fps} FPS, "
            f"{profile.clips_per_video} clip(s), and spatial size "
            f"{profile.spatial_size}. Decode batches lazily; never load the "
            "entire video corpus or split clips from one source across folds."
        )
    elif task.modality == "multimodal":
        detail = (
            "Keep all components for one entity in the same fold. Begin with "
            "late fusion: train component branches independently, produce "
            "aligned validation probabilities/embeddings using the evaluation "
            "mode selected by the harness, and let ManagerAgent fit the combiner "
            "only when compatible outputs exist. Represent missing optional inputs with explicit masks; "
            "do not discard or independently split modalities."
        )
    elif task.modality == "text":
        detail = (
            "Fit vocabulary/tokenization statistics on fold-training text only "
            "and preserve document/entity alignment."
        )
    elif task.modality == "tabular":
        detail = (
            "Use fold-fitted tabular preprocessing and preserve the current "
            "harness-owned row/fold contract."
        )
    else:
        detail = (
            f"Use fold-fitted {task.modality} preprocessing and preserve the current "
            "harness-owned sample/fold contract."
        )
    target_type = str(task.target.type or "") if task.target is not None else ""
    if target_type.endswith("_path"):
        detail += (
            f" Targets use typed `{target_type}` task-relative file references. "
            "The evaluation helper decodes selected targets after the harness "
            "split; never encode target path strings as labels or features."
        )
    if task.problem_type == "segmentation":
        detail += (
            " Decode input images lazily, preserve mask spatial alignment, train "
            "with mask tensors, restore predictions to the authoritative target "
            "resolution, and emit one N-D mask prediction per sample."
        )
    return shared + "\n" + detail
