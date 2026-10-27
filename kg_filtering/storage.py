import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_destination(path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.pending', dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        yield temporary
        with temporary.open('rb+') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_text(path, newline=None):
    with atomic_destination(path) as temporary:
        with temporary.open('w', encoding='utf-8', newline=newline) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())


def write_json_atomic(path, payload):
    with atomic_text(path) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
