from dataclasses import dataclass


@dataclass
class UsageTimer:
    """Use monotonic timestamps. Unknown intervals never count as usage."""

    delay: float = 5.0
    state: str = "unknown"
    candidate: str | None = None
    candidate_since: float = 0.0
    previous: float | None = None

    def update(self, present: bool | None, now: float) -> tuple[float, bool]:
        if self.previous is not None and now < self.previous:
            raise ValueError("timestamps must be monotonic")
        elapsed = 0.0
        # Only charge intervals bounded by valid observations.
        if self.previous is not None and present is not None and self.state == "absent":
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
