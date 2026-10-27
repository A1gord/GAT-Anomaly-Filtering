import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from .ledger import utc_timestamp


class EventJournal:
    def __init__(self, root, enabled=True):
        self.path = Path(root) / 'training_events.jsonl'
        self.enabled = enabled
        self.session = uuid4().hex
        self.started = perf_counter()
        self.lock = Lock()

    def emit(self, kind, **payload):
        if not self.enabled:
            return
        event = {'session': self.session, 'event': kind, 'timestamp': utc_timestamp(),
            'elapsed_seconds': perf_counter() - self.started, **payload}
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(encoded + '\n')
                handle.flush()
