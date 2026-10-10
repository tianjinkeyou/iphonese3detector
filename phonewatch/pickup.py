"""Estimate pickup cycles from confirmed disappearance of the parked object.

Presence cannot establish the phone's identity. Counts are evidence of observed
removal, and uncertain or unobserved intervals deliberately break a cycle.
"""

from dataclasses import dataclass, field
import math


@dataclass
class PickupTracker:
    """Debounce disappearance separately from the usage timer.

    ``update`` returns event dictionaries for a journal to timestamp and count.
    A pickup requires a previously confirmed PRESENT baseline. Startup ABSENT
    never counts as a pickup. A caller may immediately confirm a disappearance
    using a strong independent empty-area observation, so a short removal does
    not have to last for the normal debounce delay. After a long uncertain
    interval, a caller may
    explicitly resolve that baseline using a strongly confirmed empty parking
    area; the resulting pickup carries flags indicating uncertain timing.
    Short UNKNOWN intervals interrupt confirmation but may bridge a baseline;
    their total duration before the next stable observation is reported.
    """

    delay: float = 1.0
    unknown_gap: float = 2.0
    state: str = field(default="unknown", init=False)
    _previous: float | None = field(default=None, init=False, repr=False)
    _stable: str | None = field(default=None, init=False, repr=False)
    _candidate: str | None = field(default=None, init=False, repr=False)
    _candidate_since: float | None = field(default=None, init=False, repr=False)
    _unknown_since: float | None = field(default=None, init=False, repr=False)
    _unknown_seconds: float = field(default=0.0, init=False, repr=False)
    _gap_from: str | None = field(default=None, init=False, repr=False)
    _pickup_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        for name, value in (("pickup delay", self.delay), ("unknown gap", self.unknown_gap)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def _validate_time(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("timestamps must be finite")
        if self._previous is not None and now < self._previous:
            raise ValueError("timestamps must be monotonic")

    def _invalidate_gap(self, duration: float) -> None:
        if duration > self.unknown_gap and self._stable is not None:
            self._gap_from = self._stable
            self._stable = None
            self._pickup_at = None

    def update(
        self,
        present: bool | None,
        now: float,
        *,
        resolve_gap: bool = False,
        immediate_empty: bool = False,
    ) -> list[dict]:
        """Return stable transition events using finite monotonic seconds.

        Return ``away_seconds`` spans the confirmed pickup and confirmed return,
        rather than attributing any pre-confirmation interval as observed usage.
        ``confirmed_after_seconds`` records the actual confirmation delay; at a
        slow frame rate it can be greater than the configured minimum delay.
        ``resolve_gap`` is only for an independent, reliable empty-area signal.
        A recovered pickup establishes that at least one removal occurred; it
        cannot establish when it occurred or how many removals happened in the
        gap. UNKNOWN itself never produces a pickup, even with this flag.
        ``immediate_empty`` may bypass only an existing PRESENT baseline's
        absence delay, using an independently verified empty-area frame. It
        cannot bypass baseline or return confirmation. A long gap additionally
        requires ``resolve_gap``. Events that use this bypass carry
        ``rapid_empty=True``; absence shorter than one captured frame remains
        unobservable.
        """
        self._validate_time(now)
        if present is not None and not isinstance(present, bool):
            raise ValueError("presence must be True, False, or None")
        if not isinstance(resolve_gap, bool):
            raise ValueError("resolve_gap must be a boolean")
        if not isinstance(immediate_empty, bool):
            raise ValueError("immediate_empty must be a boolean")
        self._previous = now

        if present is None:
            self.state = "unknown"
            self._candidate = None
            self._candidate_since = None
            if self._unknown_since is None:
                self._unknown_since = now
            duration = self._unknown_seconds + now - self._unknown_since
            self._invalidate_gap(duration)
            return []

        if self._unknown_since is not None:
            self._unknown_seconds += now - self._unknown_since
            self._unknown_since = None
            self._invalidate_gap(self._unknown_seconds)

        target = "present" if present else "absent"
        if target == self._stable:
            self.state = target
            self._candidate = None
            self._candidate_since = None
            self._unknown_seconds = 0.0
            return []

        if target != self._candidate:
            self._candidate = target
            self._candidate_since = now
        confirmed_after = now - self._candidate_since
        rapid_empty = (
            target == "absent"
            and immediate_empty
            and confirmed_after < self.delay
            and (
                self._stable == "present"
                or (resolve_gap and self._gap_from == "present")
            )
        )
        if confirmed_after < self.delay and not rapid_empty:
            return []

        before = self._stable
        event = {
            "from_state": self._gap_from or before or "unknown",
            "to_state": target,
            "unknown_seconds": self._unknown_seconds,
            "confirmed_after_seconds": confirmed_after,
        }
        if rapid_empty:
            event["rapid_empty"] = True
        self.state = self._stable = target
        self._candidate = None
        self._candidate_since = None
        self._unknown_seconds = 0.0

        if self._gap_from is not None:
            if resolve_gap and target == "absent" and self._gap_from == "present":
                event["kind"] = "pickup"
                event["gap_recovered"] = True
                event["timing_uncertain"] = True
                # Start subsequent away time only once the area is known to be
                # empty. Do not invent the pickup's time inside the gap.
                self._pickup_at = now
            else:
                event["kind"] = "unobserved_change"
            self._gap_from = None
            return [event]
        if before is None:
            event["kind"] = "baseline"
            return [event]
        if before == "present" and target == "absent":
            self._pickup_at = now
            event["kind"] = "pickup"
            return [event]
        if self._pickup_at is not None:
            event["kind"] = "return"
            event["away_seconds"] = now - self._pickup_at
            self._pickup_at = None
            return [event]
        # An initial ABSENT baseline can become PRESENT without an observed
        # pickup. It arms subsequent removals, but is not a counted return.
        return []

    def reset(self, now: float) -> None:
        """Discard baseline and cycles when calibrating or losing the camera."""
        self._validate_time(now)
        self._previous = now
        self.state = "unknown"
        self._stable = None
        self._candidate = None
        self._candidate_since = None
        self._unknown_since = None
        self._unknown_seconds = 0.0
        self._gap_from = None
        self._pickup_at = None
