"""Versioned observational topology, independent of discovery and scheduling."""

from dataclasses import asdict, dataclass, field
import json
from math import isfinite
from types import MappingProxyType
from urllib.parse import quote


@dataclass(frozen=True)
class RelationshipType:
    directed: bool
    unit: str | None
    meaning: str


RELATIONSHIP_TYPES = MappingProxyType({
    "contains": RelationshipType(True, None, "Node contains a device or NUMA domain."),
    "nvlink": RelationshipType(False, "links", "Count of active direct GPU NVLinks; not bandwidth."),
    "pcie_ancestry": RelationshipType(False, "category", "Closest shared PCI/NUMA ancestor reported by the source."),
    "numa_locality": RelationshipType(True, "category", "Device belongs to the target NUMA domain."),
    "gpu_nic_affinity": RelationshipType(False, "category", "Derived GPU/NIC proximity; not measured performance or binding."),
})

# Ordinal topology classes only: these values are never placement costs.
AFFINITY_ORDER = (
    "same-device", "same-switch", "same-root-complex", "same-numa", "cross-numa",
)
VERTEX_KINDS = {"node", "gpu", "nic", "numa"}


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _json_copy(value):
    """Reject non-JSON data rather than silently losing keys or numeric values."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Evidence and attributes require string JSON keys")
        return {key: _json_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_copy(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError("Evidence and attributes must contain finite JSON values")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class TopologyVertex:
    kind: str
    node_id: str
    key: str
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in VERTEX_KINDS:
            raise ValueError(f"Unsupported vertex kind: {self.kind}")
        _text(self.node_id, "node_id")
        _text(self.key, "key")
        if self.kind == "node" and self.key != self.node_id:
            raise ValueError("A node vertex key must equal its node_id")
        if self.kind == "numa" and (not self.key.isascii() or not self.key.isdecimal()
                                    or str(int(self.key)) != self.key):
            raise ValueError("NUMA key must be a canonical nonnegative integer")
        if not isinstance(self.attributes, dict):
            raise ValueError("Vertex attributes must be a JSON object")
        object.__setattr__(self, "attributes", _json_copy(self.attributes))

    @property
    def id(self):
        node = f"node:{quote(self.node_id, safe='')}"
        return node if self.kind == "node" else f"{node}/{self.kind}:{quote(self.key, safe='')}"

    def as_dict(self):
        return {"id": self.id, "kind": self.kind, "node_id": self.node_id,
                "key": self.key, "attributes": _json_copy(self.attributes)}


@dataclass(frozen=True)
class TopologyRelationship:
    kind: str
    source: str
    target: str
    discovery_source: str
    value: str | int | None = None
    state: str = "known"
    confidence: str = "unknown"
    evidence: dict = field(default_factory=dict)
    reason: str | None = None

    def __post_init__(self):
        if self.kind not in RELATIONSHIP_TYPES:
            raise ValueError(f"Unsupported relationship kind: {self.kind}")
        for label in ("source", "target", "discovery_source"):
            _text(getattr(self, label), label)
        if self.source == self.target:
            raise ValueError("Self relationships are not supported")
        if self.state not in {"known", "unknown", "unsupported"}:
            raise ValueError(f"Invalid relationship state: {self.state}")
        if self.confidence not in {"high", "medium", "low", "unknown"}:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if self.reason is not None:
            _text(self.reason, "reason")
        if self.state != "known":
            if self.value is not None or not self.reason:
                raise ValueError("Unknown/unsupported relationships require null value and a reason")
        elif self.kind == "nvlink":
            if type(self.value) is not int or self.value < 0:
                raise ValueError("NVLink value must be a nonnegative integer link count")
        elif self.kind == "gpu_nic_affinity":
            if self.value not in AFFINITY_ORDER:
                raise ValueError(f"Invalid affinity category: {self.value}")
        elif self.kind == "pcie_ancestry":
            _text(self.value, "PCIe ancestry category")
        elif self.kind == "numa_locality" and self.value != "local":
            raise ValueError("NUMA locality value must be 'local'")
        elif self.kind == "contains" and self.value is not None:
            raise ValueError("Contains relationships have no numeric or categorical value")
        if not isinstance(self.evidence, dict):
            raise ValueError("Relationship evidence must be a JSON object")
        object.__setattr__(self, "evidence", _json_copy(self.evidence))
        if not RELATIONSHIP_TYPES[self.kind].directed and self.target < self.source:
            left, right = self.target, self.source
            object.__setattr__(self, "source", left)
            object.__setattr__(self, "target", right)

    @property
    def identity(self):
        return self.kind, self.source, self.target

    def as_dict(self):
        return {**asdict(self), **asdict(RELATIONSHIP_TYPES[self.kind])}


@dataclass(frozen=True)
class TopologyGraph:
    vertices: tuple[TopologyVertex, ...]
    relationships: tuple[TopologyRelationship, ...] = ()

    def __post_init__(self):
        vertices = tuple(sorted(self.vertices, key=lambda vertex: vertex.id))
        relationships = tuple(sorted(self.relationships, key=lambda edge: edge.identity))
        index = {vertex.id: vertex for vertex in vertices}
        if len(index) != len(vertices):
            raise ValueError("Duplicate topology vertex identity")
        if len({edge.identity for edge in relationships}) != len(relationships):
            raise ValueError("Duplicate relationship; combine its evidence explicitly")
        for vertex in vertices:
            owner = index.get(TopologyVertex("node", vertex.node_id, vertex.node_id).id)
            if owner is None:
                raise ValueError(f"Missing owning node vertex for {vertex.id}")
        for edge in relationships:
            if edge.source not in index or edge.target not in index:
                raise ValueError(f"Dangling relationship endpoint: {edge.identity}")
            left, right = index[edge.source], index[edge.target]
            if left.node_id != right.node_id:
                raise ValueError("These relationship types are intra-node only")
            kinds = {left.kind, right.kind}
            valid = {
                "contains": left.kind == "node" and right.kind != "node",
                "nvlink": kinds == {"gpu"},
                "pcie_ancestry": kinds <= {"gpu", "nic"},
                "numa_locality": left.kind in {"gpu", "nic"} and right.kind == "numa",
                "gpu_nic_affinity": kinds == {"gpu", "nic"},
            }[edge.kind]
            if not valid:
                raise ValueError(f"Invalid endpoint kinds for {edge.kind}")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "relationships", relationships)

    def vertex(self, identity):
        for vertex in self.vertices:
            if vertex.id == identity:
                return vertex
        raise KeyError(identity)

    def relationships_within_node(self, node_id):
        self.vertex(TopologyVertex("node", node_id, node_id).id)
        ids = {vertex.id for vertex in self.vertices if vertex.node_id == node_id}
        return tuple(edge for edge in self.relationships
                     if edge.source in ids and edge.target in ids)

    def _nearest(self, identity, expected_kind):
        vertex = self.vertex(identity)
        if vertex.kind != expected_kind:
            raise ValueError(f"Expected a {expected_kind} vertex")
        candidates = []
        for edge in self.relationships:
            if edge.kind != "gpu_nic_affinity" or edge.state != "known":
                continue
            if identity not in (edge.source, edge.target):
                continue
            peer = edge.target if edge.source == identity else edge.source
            candidates.append((AFFINITY_ORDER.index(edge.value), peer))
        if not candidates:
            return ()
        best = min(rank for rank, _ in candidates)
        return tuple(self.vertex(peer) for rank, peer in sorted(candidates) if rank == best)

    def nics_near_gpu(self, gpu_id):
        """Return all best known categorical ties, ordered by stable NIC identity."""
        return self._nearest(gpu_id, "gpu")

    def gpus_near_nic(self, nic_id):
        """Return all best known categorical ties, ordered by stable GPU identity."""
        return self._nearest(nic_id, "nic")

    def as_dict(self):
        return {"schema_version": 1,
                "vertices": [vertex.as_dict() for vertex in self.vertices],
                "relationships": [edge.as_dict() for edge in self.relationships]}

    def to_json(self):
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, data):
        """Load schema v1, or one legacy RayNodeInventory dictionary/list."""
        if isinstance(data, list) or isinstance(data, dict) and "schema_version" not in data:
            return cls.from_legacy(data)
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int \
                or data["schema_version"] != 1:
            raise ValueError("Unsupported topology schema_version")
        if set(data) != {"schema_version", "vertices", "relationships"}:
            raise ValueError("Unexpected or missing topology graph fields")
        if not isinstance(data["vertices"], list) or not isinstance(data["relationships"], list):
            raise ValueError("Topology vertices and relationships must be arrays")
        vertices = []
        for record in data["vertices"]:
            record = dict(record)
            identity = record.pop("id")
            vertex = TopologyVertex(**record)
            if identity != vertex.id:
                raise ValueError("Vertex ID does not match its kind/node/key")
            vertices.append(vertex)
        relationships = []
        for record in data["relationships"]:
            record = dict(record)
            kind = record.get("kind")
            if kind not in RELATIONSHIP_TYPES:
                raise ValueError(f"Unsupported relationship kind: {kind}")
            for key, expected in asdict(RELATIONSHIP_TYPES[kind]).items():
                if key not in record or type(record[key]) is not type(expected) or record.pop(key) != expected:
                    raise ValueError(f"Incorrect {key} for relationship {kind}")
            relationships.append(TopologyRelationship(**record))
        return cls(tuple(vertices), tuple(relationships))

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text, object_pairs_hook=_unique_object))

    @classmethod
    def from_inventory(cls, inventory):
        """Adapt one existing RayNodeInventory without modifying discovery."""
        return cls.from_legacy(asdict(inventory))

    @classmethod
    def from_observations(cls, gpu_inventory, *, nic_inventory=None, host_topology=None):
        """Join one node's GPU, NIC, and locality snapshots without running probes.

        GPU input is a RayNodeInventory or legacy dictionary. Optional inputs
        accept the dictionaries exported by NodeNICInventory/HostTopology, or
        objects with as_dict(). No unmerged collector modules are imported.
        """
        from .topology_observations import graph_from_observations

        return graph_from_observations(gpu_inventory, nic_inventory, host_topology)

    @classmethod
    def from_legacy(cls, data):
        """Retain every legacy property as attributes or per-relationship evidence."""
        records = data if isinstance(data, list) else [data]
        vertices, relationships = [], []
        for record in records:
            if not isinstance(record, dict) or not {"node_id", "devices"} <= record.keys():
                raise ValueError("Expected a legacy RayNodeInventory object with node_id and devices")
            node_id = record["node_id"]
            attributes = {key: value for key, value in record.items()
                          if key not in {"node_id", "devices", "connections"}}
            node = TopologyVertex("node", node_id, node_id, attributes)
            vertices.append(node)
            for device in record["devices"]:
                gpu = TopologyVertex("gpu", node_id, device["uuid"], device)
                vertices.append(gpu)
                relationships.append(TopologyRelationship(
                    "contains", node.id, gpu.id, "legacy-inventory"))
            for connection in record.get("connections", ()):
                left = TopologyVertex("gpu", node_id, connection["source_uuid"]).id
                right = TopologyVertex("gpu", node_id, connection["target_uuid"]).id
                for kind, key in (("nvlink", "direct_nvlink_count"),
                                  ("pcie_ancestry", "common_ancestor")):
                    value = connection.get(key)
                    unknown = value is None or (kind == "pcie_ancestry" and
                                               str(value).startswith("unknown-"))
                    relationships.append(TopologyRelationship(
                        kind, left, right, "legacy-inventory", value=None if unknown else value,
                        state="unknown" if unknown else "known", evidence=connection,
                        reason="Legacy field missing, unavailable, or unrecognized" if unknown else None,
                    ))
        return cls(tuple(vertices), tuple(relationships))
