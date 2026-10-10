from dataclasses import dataclass


@dataclass
class UsageTimer:
    """Use monotonic timestamps. Absent and unknown intervals count; present intervals do not."""

    delay: float = 5.0
    state: str = "unknown"
    candidate: str | None = None
    candidate_since: float = 0.0
    previous: float | None = None
    interval_state: str | None = None

    def update(self, present: bool | None, now: float) -> tuple[float, bool]:
        if self.previous is not None and now < self.previous:
            raise ValueError("timestamps must be monotonic")
        elapsed = 0.0
        self.interval_state = self.state if self.state in ("absent", "unknown") else None
        # Charge the interval according to the previous displayed state.
        # A new observation takes effect at now, without retroactive charging.
        if self.previous is not None and self.state in ("absent", "unknown"):
            elapsed = now - self.previous
        self.previous = now
        before = self.state
        if present is None:
            self.state = "unknown"
            self.candidate = None
        else:
            target = "present" if present else "absent"
            if target == self.state:
                self.candidate = None
            else:
                if target != self.candidate:
                    self.candidate = target
                    self.candidate_since = now
                if now - self.candidate_since >= self.delay:
                    self.state = target
                    self.candidate = None
        return elapsed, before != self.state
