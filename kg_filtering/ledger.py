import json
import os
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .provenance import digest_file
from .manifests import run_artifact_inventory
from .storage import write_json_atomic


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def study_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / '.experiment.lock'
    token = uuid4().hex
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as error:
        raise RuntimeError(f'An experiment lock already exists: {path}; inspect its owner before removing a stale lock') from error
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump({'token': token, 'pid': os.getpid(), 'created_at': utc_timestamp()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            owns_lock = path.exists() and json.loads(path.read_text(encoding='utf-8')).get('token') == token
        except (OSError, json.JSONDecodeError):
            owns_lock = False
        if owns_lock:
            path.unlink()


class RunLedger:
    def __init__(self, root):
        self.path = Path(root) / 'experiment_ledger.json'
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.is_file() else {
            'format_version': 1, 'runs': {}}
        if self.state.get('format_version') != 1 or not isinstance(self.state.get('runs'), dict):
            raise ValueError('Incompatible experiment ledger')

    def flush(self):
        write_json_atomic(self.path, self.state)

    def reusable(self, identity, signature):
        record = self.state['runs'].get(identity.key)
        if not record or record.get('status') != 'complete' or record.get('signature') != signature:
            return None
        for artifact in record['artifacts'].values():
            path = Path(artifact['path'])
            if not path.is_file() or digest_file(path) != artifact['sha256']:
                return None
        return json.loads(Path(record['artifacts']['metrics']['path']).read_text(encoding='utf-8'))

    def begin(self, identity, signature, inventory):
        self.state['runs'][identity.key] = {'identity': asdict(identity), 'signature': signature,
            'inputs': inventory, 'status': 'running', 'started_at': utc_timestamp()}
        self.flush()

    def complete(self, identity, config, result):
        artifacts = run_artifact_inventory(config, result)
        self.state['runs'][identity.key].update(status='complete', completed_at=utc_timestamp(), artifacts=artifacts)
        self.flush()

    def fail(self, identity, error):
        self.state['runs'][identity.key].update(status='failed', failed_at=utc_timestamp(),
            error={'type': type(error).__name__, 'message': str(error)})
        self.flush()
