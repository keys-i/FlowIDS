"""Causal, identity-free endpoint-ego contexts for completed flows."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

ENDPOINT_HISTORY_LIMIT = 128
MAX_HISTORY_EVENTS = 255


def _valid_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _valid_time(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class FlowEvent:
    """A completed flow's label-free routing record."""

    event_id: str
    completion_ms: int
    source_key: str
    destination_key: str

    def __post_init__(self) -> None:
        _ = _valid_text(self.event_id, "event_id")
        _ = _valid_time(self.completion_ms, "completion_ms")
        _ = _valid_text(self.source_key, "source_key")
        _ = _valid_text(self.destination_key, "destination_key")


def endpoint_relation(current: FlowEvent, earlier: FlowEvent) -> int:
    """Return the four endpoint-equality predicates packed into a 0--15 relation ID."""
    return (
        (current.source_key == earlier.source_key)
        | ((current.destination_key == earlier.destination_key) << 1)
        | ((current.source_key == earlier.destination_key) << 2)
        | ((current.destination_key == earlier.source_key) << 3)
    )


def endpoint_relation_matrix(events: Iterable[FlowEvent]) -> tuple[tuple[int, ...], ...]:
    """Return relation IDs with row event as current and column event as earlier."""
    sequence = tuple(events)
    return tuple(
        tuple(endpoint_relation(current, earlier) for earlier in sequence) for current in sequence
    )


class CausalEgoHistory:
    """Build contexts without allowing equal-time or future flows into a target history."""

    def __init__(self, horizon_ms: int, partition: str | None = None) -> None:
        self.horizon_ms: int = _valid_time(horizon_ms, "horizon_ms")
        self.partition: str | None = None
        self.reset(partition)

    def reset(self, partition: str | None = None) -> None:
        """Discard every record before processing a new capture, exporter, or split partition."""
        if partition is not None:
            _ = _valid_text(partition, "partition")
        self.partition = partition
        self._by_endpoint: defaultdict[str, deque[FlowEvent]] = defaultdict(
            lambda: deque(maxlen=ENDPOINT_HISTORY_LIMIT)
        )
        self._pending: list[FlowEvent] = []
        self._pending_time: int | None = None
        self._last_completion_ms: int | None = None
        self._seen_ids: set[str] = set()
        self._seen_order: deque[tuple[int, str]] = deque()
        self._closed: bool = False

    def add(self, target: FlowEvent) -> tuple[FlowEvent, ...]:
        """Return a target-last context and buffer the target until its timestamp closes."""
        if self._closed:
            raise ValueError("history is closed; reset it before adding another event")
        self._expire_seen(target.completion_ms)
        if target.event_id in self._seen_ids:
            raise ValueError(f"duplicate event_id within live horizon: {target.event_id}")
        if self._last_completion_ms is not None and target.completion_ms < self._last_completion_ms:
            raise ValueError("completion_ms must be nondecreasing within a partition")
        if self._pending_time is not None and target.completion_ms > self._pending_time:
            self._commit_pending()

        context = self._context(target)
        self._pending.append(target)
        self._pending_time = target.completion_ms
        self._last_completion_ms = target.completion_ms
        self._seen_ids.add(target.event_id)
        self._seen_order.append((target.completion_ms, target.event_id))
        return context

    def _expire_seen(self, completion_ms: int) -> None:
        cutoff = completion_ms - self.horizon_ms
        while self._seen_order and self._seen_order[0][0] < cutoff:
            _, event_id = self._seen_order.popleft()
            self._seen_ids.discard(event_id)

    def flush(self) -> None:
        """Commit the final timestamp batch and close the partition."""
        self._commit_pending()
        self._closed = True

    def _commit_pending(self) -> None:
        for event in sorted(self._pending, key=lambda item: item.event_id):
            self._by_endpoint[event.source_key].append(event)
            if event.destination_key != event.source_key:
                self._by_endpoint[event.destination_key].append(event)
        self._pending.clear()
        self._pending_time = None

    def _context(self, target: FlowEvent) -> tuple[FlowEvent, ...]:
        cutoff = target.completion_ms - self.horizon_ms
        candidates: dict[str, FlowEvent] = {}
        for endpoint in (target.source_key, target.destination_key):
            for event in reversed(self._by_endpoint[endpoint]):
                if event.completion_ms < cutoff:
                    break
                candidates[event.event_id] = event
        history = sorted(
            candidates.values(), key=lambda event: (event.completion_ms, event.event_id)
        )
        return tuple(history[-MAX_HISTORY_EVENTS:]) + (target,)


def replay(
    events: Iterable[FlowEvent], horizon_ms: int, partition: str | None = None
) -> tuple[tuple[FlowEvent, ...], ...]:
    """Build offline contexts by invoking the same streaming ``add`` path for every event."""
    builder = CausalEgoHistory(horizon_ms, partition)
    ordered = sorted(events, key=lambda event: (event.completion_ms, event.event_id))
    contexts = tuple(builder.add(event) for event in ordered)
    builder.flush()
    return contexts
