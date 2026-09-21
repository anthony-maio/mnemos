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
RELATED_FORGET_PREVIEW_INCOMPLETE = "related_forget_preview_incomplete"
RELATED_FORGET_PREVIEW_INCOMPLETE_MESSAGE = "The related-forget preview is incomplete. Request a larger preview and review it before applying."
RELATED_FORGET_SELECTION_CONFLICT = "related_forget_selection_conflict"
RELATED_FORGET_SELECTION_CONFLICT_MESSAGE = "The reviewed selection conflicts with source dependencies. Request a fresh preview and review the linked facts and sources."


class MnemosRejection(ValueError):
    """A request the service will not accept as sent; resending it cannot succeed.

    Clients relay only a known code with its fixed message, never exception
    text, because replies reach model context through MCP.
    """

    code = "invalid_request"
    safe_message = "The request does not match the Mnemos service contract."


class VersionConflict(MnemosRejection):
    """The caller's expected version no longer matches the stored fact."""

    code = "version_conflict"
    safe_message = "The memory changed after it was read. Read its current version before retrying."


class EventIdentityConflict(MnemosRejection):
    """A stable event identity was reused for different content."""

    code = "event_identity_conflict"
    safe_message = "This event identity was already recorded with different content."


REQUEST_REJECTIONS = {
    rejection.code: rejection.safe_message
    for rejection in (MnemosRejection, VersionConflict, EventIdentityConflict)
}


class ScopeUnavailable(ValueError):
    """The request's scope is not enabled now; the same request may succeed later.

    Registry state such as a paused project or missing workspace membership is
    configuration, not a defect in the request, so queued captures should wait
    for it instead of being dropped.
    """

    code = "scope_unavailable"


# Error codes a client may name in its own degraded message. Anything else a
# server sends is treated as opaque.
SERVICE_ERROR_CODES = frozenset(
    {
        ScopeUnavailable.code,
        "dispatch_failed",
        "dispatch_timeout",
        "job_not_found",
        "fact_not_found",
    }
)


class RelatedForgetRejection(ValueError):
    """A bounded, caller-correctable related-forget rejection."""

    code: str
    safe_message: str

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.safe_message)


class RelatedForgetSourceSelectionRequired(RelatedForgetRejection):
    """A selected related fact has a required source that was not selected."""

    code = RELATED_FORGET_SOURCE_SELECTION_REQUIRED
    safe_message = RELATED_FORGET_SOURCE_SELECTION_MESSAGE

    def __init__(self) -> None:
        super().__init__("Selected fact requires explicit source selection")


class RelatedForgetPreviewStale(RelatedForgetRejection):
    """The reviewed related-forget preview no longer matches the apply request."""

    code = RELATED_FORGET_PREVIEW_STALE
    safe_message = RELATED_FORGET_PREVIEW_STALE_MESSAGE

    def __init__(self) -> None:
        super().__init__("Forget preview is stale")


class RelatedForgetSelectionInvalid(RelatedForgetRejection):
    """A selected fact or source was not present in the reviewed preview."""

    code = RELATED_FORGET_SELECTION_INVALID
    safe_message = RELATED_FORGET_SELECTION_INVALID_MESSAGE

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Selection is not present in the reviewed forget preview")


class RelatedForgetPreviewIncomplete(RelatedForgetRejection):
    """The bounded preview omitted records required for a safe apply."""

    code = RELATED_FORGET_PREVIEW_INCOMPLETE
    safe_message = RELATED_FORGET_PREVIEW_INCOMPLETE_MESSAGE

    def __init__(self) -> None:
        super().__init__("Forget preview is incomplete")


class RelatedForgetSelectionConflict(RelatedForgetRejection):
    """The selected facts and sources have an unsafe dependency conflict."""

    code = RELATED_FORGET_SELECTION_CONFLICT
    safe_message = RELATED_FORGET_SELECTION_CONFLICT_MESSAGE

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or "Reviewed selection conflicts with source dependencies")


RELATED_FORGET_REJECTIONS = {
    RELATED_FORGET_SOURCE_SELECTION_REQUIRED: RELATED_FORGET_SOURCE_SELECTION_MESSAGE,
    RELATED_FORGET_PREVIEW_STALE: RELATED_FORGET_PREVIEW_STALE_MESSAGE,
    RELATED_FORGET_SELECTION_INVALID: RELATED_FORGET_SELECTION_INVALID_MESSAGE,
    RELATED_FORGET_PREVIEW_INCOMPLETE: RELATED_FORGET_PREVIEW_INCOMPLETE_MESSAGE,
    RELATED_FORGET_SELECTION_CONFLICT: RELATED_FORGET_SELECTION_CONFLICT_MESSAGE,
}
