import queue
import threading
import time
from typing import Callable, Dict, Optional


class BackgroundQueue:
    def __init__(self, worker: Optional[Callable[[Dict], None]] = None) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._worker = worker
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def enqueue(self, payload: Dict) -> None:
        self._queue.put(payload)

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if self._worker:
                try:
                    self._worker(payload)
                except Exception:
                    pass
            self._queue.task_done()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
