from collections import OrderedDict
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Generic, TypeVar

Key = TypeVar('Key')
Value = TypeVar('Value')


@dataclass
class CacheCounters:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversized: int = 0
    accounted_bytes: int = 0


class BudgetedCache(Generic[Key, Value]):
    def __init__(self, max_entries: int, max_bytes: int):
        if max_entries < 0 or max_bytes < 0:
            raise ValueError('Cache limits must be nonnegative')
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.entries: OrderedDict[Key, tuple[Value, int]] = OrderedDict()
        self.counters = CacheCounters()
        self.lock = RLock()

    def get(self, key: Key) -> Value | None:
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                self.counters.misses += 1
                return None
            self.entries.move_to_end(key)
            self.counters.hits += 1
            return entry[0]

    def put(self, key: Key, value: Value, cost: int) -> None:
        with self.lock:
            if cost < 0:
                raise ValueError('Cache entry cost must be nonnegative')
            if self.max_entries == 0 or self.max_bytes == 0 or cost > self.max_bytes:
                self.counters.oversized += 1
                return
            previous = self.entries.pop(key, None)
            if previous:
                self.counters.accounted_bytes -= previous[1]
            while self.entries and (len(self.entries) >= self.max_entries or
                    self.counters.accounted_bytes + cost > self.max_bytes):
                _, (_, discarded) = self.entries.popitem(last=False)
                self.counters.accounted_bytes -= discarded
                self.counters.evictions += 1
            self.entries[key] = value, cost
            self.counters.accounted_bytes += cost

    def describe(self) -> dict:
        with self.lock:
            return {**asdict(self.counters), 'entries': len(self.entries),
                'max_entries': self.max_entries, 'max_bytes': self.max_bytes,
                'accounting': 'recursive neighborhood object sizes; shared integers may be counted repeatedly'}
