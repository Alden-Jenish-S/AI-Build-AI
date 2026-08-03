"""Evidence-derived correctness constraints for generated implementations."""

from __future__ import annotations

import json

from core.contracts import TaskSpec
from evaluation.fidelity import get_fidelity_profile


def task_prompt_context(task: TaskSpec, fidelity: str) -> str:
    """Describe the resolved runtime facts without category-specific recipes.

    Representation and preprocessing decisions belong to the implementation
    agent after it has inspected the task evidence.  This function therefore
    exposes only concrete sources, targets, outputs, and resource bounds; it
    never selects behavior from a modality label.
    """
    profile = get_fidelity_profile(task.modality, fidelity)
    inputs = {
        name: {
            "role": spec.role,
            "source": spec.source,
            "format": spec.format,
            "required": spec.required,
            "options": dict(spec.options),
        }
        for name, spec in task.inputs.items()
    }
    target = task.target.to_dict() if task.target is not None else None
    output = task.output.to_dict()
    return (
        "Use the verified task evidence and runtime values exactly as observed; "
        "do not choose a loader, representation, target shape, loss, or model "
        "because of a predefined data-category label. Inspect actual values, "
        "array shapes, file signatures, and the task narrative before building "
        "the pipeline. Preserve sample/entity alignment and fit every learned "
        "operation on training data only. Validation and inference transforms "
        "must be deterministic and derived from the same fitted state.\n"
        f"Resolved inputs: {json.dumps(inputs, sort_keys=True, default=str)}\n"
        f"Resolved target: {json.dumps(target, sort_keys=True, default=str)}\n"
        f"Required output: {json.dumps(output, sort_keys=True, default=str)}\n"
        f"Evaluation limits: {json.dumps(profile.to_dict(), sort_keys=True)}"
    )


# Compatibility for external callers; the implementation is intentionally no
# longer modality-dispatched.
def modality_prompt_context(task: TaskSpec, fidelity: str) -> str:
    return task_prompt_context(task, fidelity)
