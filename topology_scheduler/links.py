"""Opt-in inter-node TCP link measurement and planner bandwidth normalization.

Measurements are time-scoped experimental evidence, not hardware facts. TCP
throughput is not RDMA, GPUDirect RDMA, NCCL, or application throughput.
"""

import errno
import json
import platform
import re
import socket
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import combinations, permutations
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

from .inventory import _node_marker
from .policy import positive

SCHEMA_VERSION = 1
SUCCEEDED, FAILED, TIMED_OUT, UNREACHABLE = (
    "succeeded", "failed", "timed_out", "unreachable")
STATUSES = (SUCCEEDED, FAILED, TIMED_OUT, UNREACHABLE)
UNITS = {
    "throughput_bytes_per_second": "bytes/s",
    "rtt_seconds": "s",
    "rtt_min_seconds": "s",
    "started_at": "Unix epoch seconds (UTC)",
    "finished_at": "Unix epoch seconds (UTC)",
    "advertised_mbps": "Mbit/s",
    "planner_bandwidth": "GB/s (decimal)",
}
# Safeguards: parameters cannot turn a probe into a long or wide load test.
MAX_DURATION_SECONDS = 60
MAX_CHUNK_BYTES = 16 << 20
MAX_STREAMS = 8
MAX_LATENCY_SAMPLES = 1000
MAX_CONCURRENT_PAIRS = 4
# The sender writes slightly past the receiver's window so that round-trip
# delay does not leave the end of the window empty.
_SEND_TAIL_SECONDS = 0.25
_MAX_ERROR_CHARS = 1000
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
_UNREACHABLE_ERRNOS = {
    getattr(errno, name) for name in ("EHOSTUNREACH", "ENETUNREACH", "EHOSTDOWN")
    if hasattr(errno, name)
}


@dataclass(frozen=True)
class ProbeParameters:
    """Bounded settings for one directional probe.

    Latency is the median of ``latency_samples`` 8-byte echo round trips on one
    connection with Nagle disabled. Throughput is what the receiver counts
    across ``streams`` parallel connections after ``warmup_seconds``, for
    ``duration_seconds``, while the sender writes ``chunk_bytes`` at a time.
    ``max_concurrent_pairs`` bounds how many directions run at once; directions
    in one batch never share a node.
    """

    duration_seconds: float = 5.0
    warmup_seconds: float = 1.0
    chunk_bytes: int = 1 << 20
    streams: int = 1
    latency_samples: int = 20
    timeout_seconds: float = 30.0
    max_concurrent_pairs: int = 1

    def __post_init__(self):
        positive(self.duration_seconds, "duration_seconds")
        positive(self.warmup_seconds, "warmup_seconds", zero=True)
        positive(self.timeout_seconds, "timeout_seconds")
        if self.duration_seconds > MAX_DURATION_SECONDS:
            raise ValueError(f"duration_seconds must not exceed {MAX_DURATION_SECONDS}")
        for name, upper in (
            ("chunk_bytes", MAX_CHUNK_BYTES),
            ("streams", MAX_STREAMS),
            ("latency_samples", MAX_LATENCY_SAMPLES),
            ("max_concurrent_pairs", MAX_CONCURRENT_PAIRS),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer from 1 to {upper}")
        if self.timeout_seconds <= self.warmup_seconds + self.duration_seconds + 1:
            raise ValueError(
                "timeout_seconds must exceed warmup_seconds + duration_seconds + 1"
            )

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LinkEndpoint:
    """One side of a probe; interface fields are ``None`` when unavailable.

    On Linux the interface owning ``address`` is resolved, and its advertised
    speed is read from sysfs. Elsewhere, or for virtual interfaces without a
    reported speed, both stay ``None``.
    """

    node_name: str
    node_id: str
    address: str
    interface: str | None = None
    advertised_mbps: float | None = None


@dataclass(frozen=True)
class LinkMeasurement:
    """Raw evidence for one direction, from ``source`` to ``destination``.

    Times come from the source node's clock when the probe ran there, and from
    the driver's clock when it never started.
    """

    source: LinkEndpoint
    destination: LinkEndpoint
    status: str
    started_at: float
    finished_at: float
    parameters: ProbeParameters
    throughput_bytes_per_second: float | None = None
    rtt_seconds: float | None = None
    rtt_min_seconds: float | None = None
    error: str | None = None
    hint: str | None = None
    software: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of: {', '.join(STATUSES)}")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.status == SUCCEEDED:
            if self.throughput_bytes_per_second is None or self.rtt_seconds is None:
                raise ValueError("A successful measurement needs throughput and RTT")
            positive(self.throughput_bytes_per_second, "throughput_bytes_per_second")
            positive(self.rtt_seconds, "rtt_seconds", zero=True)
        elif not self.error:
            raise ValueError("An unsuccessful measurement needs an error")

    @property
    def direction(self) -> str:
        return f"{self.source.node_name}->{self.destination.node_name}"

    def as_dict(self) -> dict:
        return {
            "direction": self.direction,
            "source": asdict(self.source),
            "destination": asdict(self.destination),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "started_at_utc": _utc(self.started_at),
            "parameters": self.parameters.as_dict(),
            "throughput_bytes_per_second": self.throughput_bytes_per_second,
            "rtt_seconds": self.rtt_seconds,
            "rtt_min_seconds": self.rtt_min_seconds,
            "error": self.error,
            "hint": self.hint,
            "software": dict(self.software),
            "units": dict(UNITS),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "LinkMeasurement":
        return cls(
            source=LinkEndpoint(**value["source"]),
            destination=LinkEndpoint(**value["destination"]),
            status=value["status"],
            started_at=value["started_at"],
            finished_at=value["finished_at"],
            parameters=ProbeParameters(**value["parameters"]),
            throughput_bytes_per_second=value.get("throughput_bytes_per_second"),
            rtt_seconds=value.get("rtt_seconds"),
            rtt_min_seconds=value.get("rtt_min_seconds"),
            error=value.get("error"),
            hint=value.get("hint"),
            software=tuple(sorted(value.get("software", {}).items())),
        )


@dataclass(frozen=True)
class LinkMeasurementReport:
    """Directional measurements from one or more probe runs."""

    measurements: tuple[LinkMeasurement, ...]

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "measurements": [item.as_dict() for item in self.measurements],
        }

    def save(self, path) -> None:
        """Write the report as JSON for later reuse under an explicit TTL."""
        Path(path).write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_link_report(path) -> LinkMeasurementReport:
    """Read a report written by ``LinkMeasurementReport.save``."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported link report schema {value.get('schema_version')!r}")
    return LinkMeasurementReport(
        tuple(LinkMeasurement.from_dict(item) for item in value["measurements"]))


class LinkCostSource(str, Enum):
    """Where a planner bandwidth value came from."""

    MEASURED = "measured"
    ADVERTISED = "advertised"
    FALLBACK = "fallback"
    SUPPLIED = "supplied"


@dataclass(frozen=True)
class LinkCost:
    """One undirected planner bandwidth value in GB/s and its provenance."""

    gb_per_second: float
    source: LinkCostSource
    measured_at: float | None = None

    def __post_init__(self):
        positive(self.gb_per_second, "gb_per_second")
        object.__setattr__(self, "source", LinkCostSource(self.source))

    def provenance(self) -> dict:
        return {"source": self.source.value, "measured_at": self.measured_at}


@dataclass(frozen=True)
class LinkResolution:
    """Normalized planner costs plus a diagnostic for every non-measured pair."""

    costs: Mapping[tuple[str, str], LinkCost]
    diagnostics: tuple[str, ...]

    @property
    def bandwidth_gbps(self) -> dict[tuple[str, str], float]:
        """The map accepted by ``choose_placement`` and ``plan_with_record``."""
        return {pair: cost.gb_per_second for pair, cost in sorted(self.costs.items())}


def resolve_link_costs(
    report: LinkMeasurementReport | None,
    node_names: Iterable[str], *,
    max_age_seconds: float,
    now: float | None = None,
    use_advertised: bool = False,
    fallback_gb_per_second: float | None = None,
) -> LinkResolution:
    """Normalize directional evidence into the planner's undirected GB/s map.

    A pair is ``measured`` only when the newest record in each direction
    succeeded within ``max_age_seconds``. Its value is the slower direction,
    because the planner charges one symmetric cost per pair. Otherwise the pair
    uses the slower advertised endpoint speed when ``use_advertised`` is set and
    both speeds are known, then ``fallback_gb_per_second`` when given. If
    neither applies the pair is omitted, so placements that communicate across
    it stay infeasible. Every non-measured pair gets a diagnostic; stale
    evidence is never reused as a measurement.
    """
    positive(max_age_seconds, "max_age_seconds")
    if fallback_gb_per_second is not None:
        positive(fallback_gb_per_second, "fallback_gb_per_second")
    now = time.time() if now is None else now
    latest: dict[tuple[str, str], LinkMeasurement] = {}
    for item in report.measurements if report is not None else ():
        key = (item.source.node_name, item.destination.node_name)
        if key not in latest or item.finished_at > latest[key].finished_at:
            latest[key] = item

    costs = {}
    diagnostics = []
    for left, right in combinations(sorted(set(node_names)), 2):
        usable, problems = [], []
        for source, destination in ((left, right), (right, left)):
            item = latest.get((source, destination))
            label = f"{source}->{destination}"
            if item is None:
                problems.append(f"{label} has no measurement")
            elif item.status != SUCCEEDED:
                problems.append(f"{label} {item.status}: {item.error}"
                                + (f" Hint: {item.hint}" if item.hint else ""))
            elif now - item.finished_at > max_age_seconds:
                problems.append(f"{label} is stale ({now - item.finished_at:.0f}s old; "
                                f"max_age_seconds={max_age_seconds:g})")
            else:
                usable.append(item)
        if len(usable) == 2:
            costs[(left, right)] = LinkCost(
                min(item.throughput_bytes_per_second for item in usable) / 1e9,
                LinkCostSource.MEASURED,
                min(item.finished_at for item in usable),
            )
            continue
        advertised = _advertised_gb_per_second(
            latest.get((left, right)), latest.get((right, left)))
        if use_advertised and advertised is not None:
            costs[(left, right)] = LinkCost(advertised, LinkCostSource.ADVERTISED)
            used = "advertised capacity"
        elif fallback_gb_per_second is not None:
            costs[(left, right)] = LinkCost(fallback_gb_per_second, LinkCostSource.FALLBACK)
            used = "the fallback value"
        else:
            used = "no value, so placements communicating across it are infeasible"
        diagnostics.append(
            f"{left}|{right} is not measured ({'; '.join(problems)}); using {used}. "
            "Re-measure this pair with measure_ray_links() to replace it."
        )
    return LinkResolution(costs, tuple(diagnostics))


def _advertised_gb_per_second(*items: LinkMeasurement | None) -> float | None:
    """Return the slower advertised endpoint speed of one pair in GB/s."""
    speeds: dict[str, float] = {}
    for item in items:
        if item is None:
            continue
        for endpoint in (item.source, item.destination):
            if endpoint.advertised_mbps is not None:
                speeds[endpoint.node_name] = min(
                    speeds.get(endpoint.node_name, endpoint.advertised_mbps),
                    endpoint.advertised_mbps,
                )
    if len(speeds) != 2:
        return None
    return min(speeds.values()) / 8000  # Mbit/s -> decimal GB/s


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _interface_for_address(address: str) -> str | None:
    """Find the Linux interface whose primary IPv4 address is ``address``."""
    if not sys.platform.startswith("linux"):
        return None
    import fcntl

    siocgifaddr = 0x8915
    try:
        names = [name for _, name in socket.if_nameindex()]
    except OSError:
        return None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for name in names:
            try:
                reply = fcntl.ioctl(
                    sock.fileno(), siocgifaddr, struct.pack("256s", name.encode()[:15]))
            except OSError:
                continue
            if socket.inet_ntoa(reply[20:24]) == address:
                return name
    return None


def _advertised_mbps(interface: str | None) -> float | None:
    """Read the link speed Linux reports for an interface, when it has one."""
    if interface is None:
        return None
    try:
        value = float(Path("/sys/class/net", interface, "speed").read_text().strip())
    except (OSError, ValueError):
        return None
    return value if value > 0 else None


def _endpoint(node_name: str, node_id: str, address: str) -> LinkEndpoint:
    interface = _interface_for_address(address)
    return LinkEndpoint(node_name, node_id, address, interface, _advertised_mbps(interface))


def _route_address(peer: str) -> str:
    """Return the local address the kernel would use to reach ``peer``."""
    family, kind, _, _, target = socket.getaddrinfo(peer, 9, type=socket.SOCK_DGRAM)[0]
    with socket.socket(family, kind) as sock:
        sock.connect(target)  # A UDP connect selects a route without sending.
        return sock.getsockname()[0]


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("peer closed the connection mid-message")
        data.extend(chunk)
    return bytes(data)


def _shutdown(sock: socket.socket) -> None:
    """Close a socket in a way that also wakes a thread blocked on it."""
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


class LinkProbeServer:
    """Serve one latency session and ``streams`` throughput connections.

    The listener waits up to ``timeout_seconds`` for the first connection; the
    rest of the session must then finish within another ``timeout_seconds``.
    """

    def __init__(self, parameters: ProbeParameters, bind_address: str):
        self.parameters = parameters
        family = socket.AF_INET6 if ":" in bind_address else socket.AF_INET
        self._listener = socket.create_server(
            (bind_address, 0), family=family, backlog=parameters.streams + 1)
        self.port = self._listener.getsockname()[1]
        self._received = [0] * parameters.streams
        self._errors: list[str] = []
        self._connections: list[socket.socket] = []
        self._done = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self) -> None:
        """Stop waiting for a client that is known to have failed."""
        _shutdown(self._listener)
        for conn in list(self._connections):
            _shutdown(conn)

    def result(self) -> dict:
        """Wait for the session and return the receiver's throughput."""
        if not self._done.wait(2 * self.parameters.timeout_seconds + 1):
            self._errors.append("probe server did not finish")
        errors = list(self._errors)
        throughput = (
            None if errors else sum(self._received) / self.parameters.duration_seconds)
        return {"throughput_bytes_per_second": throughput, "errors": errors}

    def _serve(self) -> None:
        timeout = self.parameters.timeout_seconds
        workers: list[threading.Thread] = []
        deadline = None
        try:
            self._listener.settimeout(timeout)
            for _ in range(self.parameters.streams + 1):
                conn, _ = self._listener.accept()
                if deadline is None:
                    deadline = time.monotonic() + timeout
                conn.settimeout(timeout)
                self._connections.append(conn)
                worker = threading.Thread(target=self._handle, args=(conn,), daemon=True)
                worker.start()
                workers.append(worker)
                self._listener.settimeout(max(deadline - time.monotonic(), 0.001))
            for worker in workers:
                worker.join(max(deadline - time.monotonic(), 0))
            if any(worker.is_alive() for worker in workers):
                self._errors.append(
                    f"session did not finish within timeout_seconds={timeout:g}")
        except OSError as error:
            self._errors.append(f"server {type(error).__name__}: {error}")
        finally:
            self.close()
            for worker in workers:
                worker.join(1)
            self._done.set()

    def _handle(self, conn: socket.socket) -> None:
        try:
            mode, index = _recv_exact(conn, 2)
            if mode == ord("L"):
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    first = conn.recv(8)
                    if not first:
                        return
                    conn.sendall(first + _recv_exact(conn, 8 - len(first)))
            elif mode == ord("T") and index < self.parameters.streams:
                self._count(conn, index)
            else:
                raise ValueError(f"unexpected probe header {bytes((mode, index))!r}")
        except (OSError, ValueError) as error:
            self._errors.append(f"server {type(error).__name__}: {error}")

    def _count(self, conn: socket.socket, index: int) -> None:
        """Count bytes that arrive inside this stream's measurement window."""
        buffer = memoryview(bytearray(min(self.parameters.chunk_bytes, 1 << 20)))
        begin = end = last = None
        while True:
            size = conn.recv_into(buffer)
            now = time.perf_counter()
            if not size:
                break
            if begin is None:
                begin = now + self.parameters.warmup_seconds
                end = begin + self.parameters.duration_seconds
            if begin <= now <= end:
                self._received[index] += size
            last = now
        if last is None or last < end:
            raise ConnectionError(
                f"stream {index} closed before its measurement window ended")


def _send_until(stream: socket.socket, chunk: bytes, deadline: float,
                errors: list) -> None:
    try:
        while time.perf_counter() < deadline:
            stream.sendall(chunk)
    except OSError as error:
        errors.append(error)


def run_link_client(address: str, port: int, parameters: ProbeParameters) -> dict:
    """Measure RTT, then send throughput streams to a ``LinkProbeServer``."""
    timeout = parameters.timeout_seconds
    samples = []
    with socket.create_connection((address, port), timeout=timeout) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.sendall(b"L\0")
        for sample in range(parameters.latency_samples):
            payload = struct.pack("!Q", sample)
            started = time.perf_counter()
            conn.sendall(payload)
            if _recv_exact(conn, 8) != payload:
                raise ConnectionError("latency echo did not match what was sent")
            samples.append(time.perf_counter() - started)

    streams: list[socket.socket] = []
    errors: list[OSError] = []
    try:
        for index in range(parameters.streams):
            streams.append(socket.create_connection((address, port), timeout=timeout))
            streams[-1].sendall(b"T" + bytes((index,)))
        deadline = (time.perf_counter() + parameters.warmup_seconds
                    + parameters.duration_seconds + _SEND_TAIL_SECONDS)
        chunk = bytes(parameters.chunk_bytes)
        senders = [
            threading.Thread(target=_send_until, args=(stream, chunk, deadline, errors))
            for stream in streams
        ]
        for sender in senders:
            sender.start()
        for sender in senders:
            sender.join()
    finally:
        for stream in streams:
            try:
                stream.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            stream.close()
    if errors:
        raise errors[0]
    return {"rtt_seconds": median(samples), "rtt_min_seconds": min(samples)}


def _attempt(function, *args) -> tuple:
    try:
        return function(*args), None
    except Exception as error:
        return None, error


class ProbeServerStartError(RuntimeError):
    """The destination probe server could not be started."""


def _describe(error: BaseException) -> str:
    """Render an error without terminal colors, keeping the root cause.

    Ray errors end with the original traceback, so an overlong message keeps
    its tail rather than being cut at the front only.
    """
    text = _ANSI_ESCAPE.sub("", f"{type(error).__name__}: {error}")
    if len(text) <= _MAX_ERROR_CHARS:
        return text
    head = _MAX_ERROR_CHARS // 3
    return f"{text[:head]} ... {text[head + 5 - _MAX_ERROR_CHARS:]}"


def _classify(error: BaseException, source: str, destination: str,
              address: str) -> tuple[str, str]:
    """Map a probe failure to a status and an actionable hint."""
    if isinstance(error, ProbeServerStartError):
        return FAILED, (
            f"The probe server on {destination} could not listen on {address}. "
            f"Confirm that this Ray node address belongs to an interface on "
            f"{destination} and that this package is installed there."
        )
    if isinstance(error, TimeoutError):
        return TIMED_OUT, (
            f"No progress from {source} to {destination} within timeout_seconds. "
            "Check for packet loss or congestion, confirm neither node is "
            "overloaded, or raise timeout_seconds."
        )
    if (isinstance(error, ConnectionRefusedError)
            or getattr(error, "errno", None) in _UNREACHABLE_ERRNOS):
        return UNREACHABLE, (
            f"{source} could not open TCP to {destination} at {address}. Allow "
            "ephemeral TCP ports between these nodes and confirm that Ray's node "
            "address is routable from the source."
        )
    return FAILED, (
        f"Rerun {source}->{destination} alone with a short duration_seconds to "
        "isolate the failure, and include this error when reporting it."
    )


def _software() -> tuple[tuple[str, str], ...]:
    from importlib.metadata import PackageNotFoundError, version

    values = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("topology-aware-gpu-scheduling", "ray"):
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "not installed"
    return tuple(sorted(values.items()))


def _measurement(
    source: LinkEndpoint, destination: LinkEndpoint, parameters: ProbeParameters,
    started_at: float, finished_at: float, client: tuple, server: tuple,
    software: tuple[tuple[str, str], ...],
) -> LinkMeasurement:
    """Combine ``(result, error)`` client and server outcomes into one record."""
    (client_result, client_error), (server_result, server_error) = client, server
    error = client_error or server_error
    if error is None and server_result["errors"]:
        error = ConnectionError("; ".join(server_result["errors"]))
    if error is None and not server_result["throughput_bytes_per_second"]:
        error = ConnectionError("no bytes arrived inside the measurement window")
    common = dict(
        source=source, destination=destination, started_at=started_at,
        finished_at=max(finished_at, started_at), parameters=parameters,
        software=software,
    )
    if error is not None:
        status, hint = _classify(
            error, source.node_name, destination.node_name, destination.address)
        return LinkMeasurement(status=status, error=_describe(error), hint=hint, **common)
    return LinkMeasurement(
        status=SUCCEEDED,
        throughput_bytes_per_second=server_result["throughput_bytes_per_second"],
        rtt_seconds=client_result["rtt_seconds"],
        rtt_min_seconds=client_result["rtt_min_seconds"],
        **common,
    )


def measure_loopback_link(parameters: ProbeParameters | None = None) -> LinkMeasurement:
    """Probe this host over 127.0.0.1 without Ray.

    This exercises the TCP probe end to end. The result describes the local
    loopback path and must never be used as a cluster link cost.
    """
    parameters = parameters or ProbeParameters(
        duration_seconds=0.5, warmup_seconds=0.1, latency_samples=5, timeout_seconds=10)
    address = "127.0.0.1"
    started = time.time()
    server = LinkProbeServer(parameters, address)
    client = _attempt(run_link_client, address, server.port, parameters)
    if client[1] is not None:
        server.close()
    return _measurement(
        _endpoint("loopback-source", "local", address),
        _endpoint("loopback-destination", "local", address),
        parameters, started, time.time(), client, (server.result(), None), _software(),
    )


class _ServerActor:
    """Ray actor body that hosts one probe server on the destination node."""

    def __init__(self, parameters: ProbeParameters, address: str):
        self._server = LinkProbeServer(parameters, address)

    def ready(self, node_name: str, address: str) -> dict:
        import ray

        node_id = ray.get_runtime_context().get_node_id()
        return {"endpoint": _endpoint(node_name, node_id, address),
                "port": self._server.port}

    def close(self) -> None:
        self._server.close()

    def result(self) -> dict:
        return self._server.result()


def _client_task(node_name: str, address: str, port: int,
                 parameters: ProbeParameters) -> dict:
    """Ray task body pinned to the source node."""
    import ray

    node_id = ray.get_runtime_context().get_node_id()
    try:
        endpoint = _endpoint(node_name, node_id, _route_address(address))
    except OSError:
        endpoint = LinkEndpoint(node_name, node_id, "unknown")
    started = time.time()
    result, error = _attempt(run_link_client, address, port, parameters)
    return {"endpoint": endpoint, "started_at": started, "finished_at": time.time(),
            "result": result, "error": error, "software": _software()}


def _live_topology_nodes(nodes: list[dict]) -> dict[str, tuple[str, str]]:
    """Map topology names to ``(Ray node ID, node address)`` for live nodes."""
    live = {}
    for node in nodes:
        resources = node["Resources"]
        if not node["Alive"] or not any(
                key.startswith("topology_node:") and quantity > 0
                for key, quantity in resources.items()):
            continue
        _, name = _node_marker(resources, node["NodeID"])
        if name in live:
            raise ValueError(
                f"topology_node:{name} is advertised by more than one live Ray node")
        live[name] = (node["NodeID"], node["NodeManagerAddress"])
    return live


def _disjoint_batches(pairs: list[tuple[str, str]],
                      limit: int) -> list[list[tuple[str, str]]]:
    """Group directions so that no node takes part in two probes at once."""
    batches, remaining = [], list(pairs)
    while remaining:
        batch, busy = [], set()
        for pair in remaining:
            if len(batch) < limit and busy.isdisjoint(pair):
                batch.append(pair)
                busy.update(pair)
        remaining = [pair for pair in remaining if pair not in batch]
        batches.append(batch)
    return batches


def _gather(ray, refs: list, timeout: float, stage: str) -> list[tuple]:
    """Collect ``(value, error)`` per ref so one failure hides no others."""
    if not refs:
        return []
    done, _ = ray.wait(refs, num_returns=len(refs), timeout=timeout)
    done = set(done)
    outcomes = []
    for ref in refs:
        if ref not in done:
            outcomes.append(
                (None, TimeoutError(f"{stage} returned nothing within {timeout:g}s")))
            continue
        try:
            outcomes.append((ray.get(ref), None))
        except Exception as error:
            outcomes.append((None, error))
    return outcomes


def _measure_batch(ray, batch, live, parameters, server_actor, client_task,
                   pinned) -> list[LinkMeasurement]:
    """Run one batch of node-disjoint directions and record each outcome."""
    timeout = parameters.timeout_seconds
    batch_started = time.time()
    servers = [
        server_actor.options(scheduling_strategy=pinned(live[destination][0]))
        .remote(parameters, live[destination][1])
        for _, destination in batch
    ]
    client_refs = {}
    try:
        ready = [
            (info, error if error is None or isinstance(error, TimeoutError)
             else ProbeServerStartError(_describe(error)))
            for info, error in _gather(ray, [
                server.ready.remote(destination, live[destination][1])
                for server, (_, destination) in zip(servers, batch)
            ], timeout, "destination probe server")
        ]
        for index, ((source, destination), (info, error)) in enumerate(zip(batch, ready)):
            if error is None:
                client_refs[index] = client_task.options(
                    scheduling_strategy=pinned(live[source][0]),
                ).remote(source, live[destination][1], info["port"], parameters)
        clients = dict(zip(client_refs, _gather(
            ray, list(client_refs.values()), 2 * timeout, "source probe client")))
        for index, (task, task_error) in clients.items():
            if task_error is not None or task["error"] is not None:
                servers[index].close.remote()
        result_refs = {index: servers[index].result.remote() for index in client_refs}
        results = dict(zip(result_refs, _gather(
            ray, list(result_refs.values()), 2 * timeout + 5,
            "destination probe server")))

        measurements = []
        for index, (source, destination) in enumerate(batch):
            info, ready_error = ready[index]
            task, task_error = clients.get(index, (None, ready_error))
            task = task or {}
            measurements.append(_measurement(
                task.get("endpoint") or LinkEndpoint(source, *live[source]),
                (info or {}).get("endpoint") or LinkEndpoint(destination, *live[destination]),
                parameters,
                task.get("started_at", batch_started),
                task.get("finished_at", time.time()),
                (task.get("result"), task.get("error") or task_error),
                results.get(index, (None, ready_error)),
                task.get("software") or _software(),
            ))
        return measurements
    finally:
        # A client that outlived its timeout may still be sending; stop it.
        for ref in client_refs.values():
            ray.cancel(ref, force=True)
        for server in servers:
            ray.kill(server)


def measure_ray_links(
    node_names: Iterable[str] | None = None, *,
    parameters: ProbeParameters | None = None,
    allow_network_load: bool = False,
) -> LinkMeasurementReport:
    """Probe both directions of every pair among live Ray topology nodes.

    This is opt-in and never runs during planning. It sends real TCP traffic
    between the selected nodes, so ``allow_network_load=True`` is required and
    should only be passed inside a controlled measurement window. Each node must
    advertise one ``topology_node:<name>`` resource; its Ray node address is the
    probe target. Failed directions are recorded in the report, not raised.
    """
    if allow_network_load is not True:
        raise ValueError(
            "measure_ray_links sends real network traffic; pass "
            "allow_network_load=True only during a controlled measurement window"
        )
    parameters = parameters or ProbeParameters()
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    if not ray.is_initialized():
        raise RuntimeError("Call ray.init() before measuring links")
    live = _live_topology_nodes(ray.nodes())
    selected = sorted(live) if node_names is None else sorted(set(node_names))
    missing = [name for name in selected if name not in live]
    if missing:
        raise ValueError(
            "No live Ray node advertises "
            + ", ".join(f"topology_node:{name}" for name in missing))
    if len(selected) < 2:
        raise ValueError("Select at least two topology nodes to measure")

    def pinned(node_id):
        return NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)

    server_actor = ray.remote(num_cpus=0, max_restarts=0)(_ServerActor)
    client_task = ray.remote(num_cpus=0, max_retries=0)(_client_task)
    measurements = []
    for batch in _disjoint_batches(
            list(permutations(selected, 2)), parameters.max_concurrent_pairs):
        measurements.extend(_measure_batch(
            ray, batch, live, parameters, server_actor, client_task, pinned))
    return LinkMeasurementReport(tuple(measurements))
