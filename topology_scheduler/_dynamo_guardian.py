"""Linux subprocess guardian: EOF/lease expiry stops owned descendants.

Private entry point. The actor supplies one JSON launch record, then heartbeat
lines over stdin. The engine never inherits that pipe. A receipt is written
only after all descendants are reaped, including children that change session.
"""

import ctypes
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time


def _processes():
    result = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            result[int(path.parent.name)] = (int(fields[1]), fields[19])
        except (OSError, ValueError, IndexError):
            continue
    return result


def descendants():
    processes = _processes()
    owned = {os.getpid()}
    while True:
        children = {pid for pid, (parent, _) in processes.items() if parent in owned}
        if children <= owned:
            break
        owned |= children
    return {pid: processes[pid][1] for pid in owned if pid != os.getpid()}


def _signal_owned(sig):
    for pid, birth in descendants().items():
        # Avoid signaling a recycled PID from a previous /proc snapshot.
        if _processes().get(pid, (None, None))[1] == birth:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass


def _reap():
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                return False
        except ChildProcessError:
            return True


def stop_tree(grace, kill_wait):
    for sig, duration in ((signal.SIGTERM, grace), (signal.SIGKILL, kill_wait)):
        deadline = time.monotonic() + duration
        while True:
            _signal_owned(sig)
            no_children = _reap()
            if no_children and not descendants():
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
    return False


def main():
    if sys.platform != "linux":
        raise RuntimeError("Dynamo process containment requires Linux /proc")
    # Adopt grandchildren when an intermediate engine process exits.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    # Read the launch line without buffering subsequent heartbeat bytes.
    launch = bytearray()
    while not launch.endswith(b"\n"):
        byte = os.read(0, 1)
        if not byte:
            return 1
        launch.extend(byte)
    settings = json.loads(launch)
    child = None
    reason = "startup failed"
    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        child = subprocess.Popen(settings["command"], stdin=subprocess.DEVNULL,
                                 close_fds=True, start_new_session=True)
        deadline = time.monotonic() + settings["lease_timeout"]
        with selectors.DefaultSelector() as selector:
            selector.register(0, selectors.EVENT_READ)
            while not stopping:
                code = child.poll()
                if code is not None:
                    reason = f"engine exited with code {code}"
                    break
                if time.monotonic() >= deadline:
                    reason = "driver heartbeat expired"
                    break
                if selector.select(timeout=min(0.1, max(0, deadline - time.monotonic()))):
                    message = os.read(0, 4096)
                    if not message or b"stop" in message:
                        reason = "owner disconnected or requested stop"
                        break
                    deadline = time.monotonic() + settings["lease_timeout"]
            else:
                reason = "guardian received shutdown signal"
    finally:
        stopped = stop_tree(settings["shutdown_timeout"], settings["kill_timeout"])
        receipt = Path(settings["receipt"])
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps({"token": settings["token"], "stopped": stopped,
                                         "reason": reason}), encoding="utf-8")
        temporary.replace(receipt)
    return 0 if stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
