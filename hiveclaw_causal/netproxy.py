"""Userspace TCP proxy: delay and drop connections without killing client processes."""

from __future__ import annotations

import select
import socket
import threading
import time
from collections.abc import Callable


class TcpProxy:
    """Listen on 127.0.0.1:ephemeral and forward to dest_host:dest_port."""

    def __init__(self, dest_host: str, dest_port: int, *, delay_s: float = 0.0) -> None:
        self.dest_host = dest_host
        self.dest_port = dest_port
        self.delay_s = delay_s
        self.port = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listen: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._pipes: list[socket.socket] = []
        self._stalled = threading.Event()

    def start(self) -> int:
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen.bind(("127.0.0.1", 0))
        listen.listen(128)
        listen.settimeout(0.2)
        self._listen = listen
        self.port = int(listen.getsockname()[1])
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        return self.port

    def set_delay(self, delay_s: float) -> None:
        self.delay_s = delay_s

    def stall(self, seconds: float) -> None:
        """Stop forwarding for `seconds` without closing sockets (jitter / GC analogue)."""
        self._stalled.set()
        try:
            time.sleep(seconds)
        finally:
            self._stalled.clear()

    def drop_all(self) -> None:
        with self._lock:
            socks = list(self._pipes)
            self._pipes.clear()
        for s in socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        self.drop_all()
        if self._listen is not None:
            try:
                self._listen.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _track(self, *socks: socket.socket) -> None:
        with self._lock:
            self._pipes.extend(socks)

    def _untrack(self, *socks: socket.socket) -> None:
        with self._lock:
            for s in socks:
                if s in self._pipes:
                    self._pipes.remove(s)

    def _accept_loop(self) -> None:
        assert self._listen is not None
        while not self._stop.is_set():
            try:
                client, _addr = self._listen.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(client,), daemon=True).start()

    def _handle(self, client: socket.socket) -> None:
        delay = self.delay_s
        if delay > 0:
            time.sleep(delay)
        try:
            server = socket.create_connection((self.dest_host, self.dest_port), timeout=8)
        except OSError:
            try:
                client.close()
            except OSError:
                pass
            return
        self._track(client, server)
        try:
            _relay(client, server, lambda: self.delay_s, self._stalled)
        finally:
            self._untrack(client, server)
            for s in (client, server):
                try:
                    s.close()
                except OSError:
                    pass


def _relay(
    a: socket.socket,
    b: socket.socket,
    delay_fn: Callable[[], float],
    stalled: threading.Event,
) -> None:
    a.setblocking(False)
    b.setblocking(False)
    sockets = [a, b]
    try:
        while True:
            while stalled.is_set():
                time.sleep(0.01)
            readable, _, exceptional = select.select(sockets, [], sockets, 0.2)
            if exceptional:
                return
            if not readable:
                continue
            delay = delay_fn()
            for src in readable:
                dst = b if src is a else a
                try:
                    data = src.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                if delay > 0:
                    time.sleep(delay)
                while stalled.is_set():
                    time.sleep(0.01)
                try:
                    dst.sendall(data)
                except OSError:
                    return
    except (OSError, ValueError):
        return
