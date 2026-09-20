"""Stable, narrow domain errors shared by the service and authenticated clients."""

from __future__ import annotations

RELATED_FORGET_SOURCE_SELECTION_REQUIRED = "related_forget_source_selection_required"
RELATED_FORGET_SOURCE_SELECTION_MESSAGE = (
    "Review and select every required source event before applying related forget."
)
RELATED_FORGET_PREVIEW_STALE = "related_forget_preview_stale"
RELATED_FORGET_PREVIEW_STALE_MESSAGE = (
    "The related-forget preview is stale. Request a fresh preview and reuse its digest and limit."
)
RELATED_FORGET_SELECTION_INVALID = "related_forget_selection_invalid"
RELATED_FORGET_SELECTION_INVALID_MESSAGE = (
    "Selected related-forget IDs must come from the reviewed preview."
)


class RelatedForgetSourceSelectionRequired(ValueError):
    """A selected related fact has a required source that was not selected."""

    code = RELATED_FORGET_SOURCE_SELECTION_REQUIRED
    safe_message = RELATED_FORGET_SOURCE_SELECTION_MESSAGE

    def __init__(self) -> None:
        super().__init__("Selected fact requires explicit source selection")


class RelatedForgetPreviewStale(ValueError):
    """The reviewed related-forget preview no longer matches the apply request."""

    code = RELATED_FORGET_PREVIEW_STALE
    safe_message = RELATED_FORGET_PREVIEW_STALE_MESSAGE

    def __init__(self) -> None:
        super().__init__("Forget preview is stale")


class RelatedForgetSelectionInvalid(ValueError):
    """A selected fact or source was not present in the reviewed preview."""

    code = RELATED_FORGET_SELECTION_INVALID
    safe_message = RELATED_FORGET_SELECTION_INVALID_MESSAGE

    def __init__(self) -> None:
        super().__init__("Selection is not present in the reviewed forget preview")
