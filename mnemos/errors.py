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
