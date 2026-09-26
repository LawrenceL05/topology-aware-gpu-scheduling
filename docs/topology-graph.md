# Typed topology graph

`TopologyGraph` represents GPU, NIC, NUMA, and node entities with separate
NVLink, PCI/NUMA ancestry, NUMA membership, and GPU-to-NIC affinity relationships.
It does not collapse those observations into one numeric distance or cost.

The graph API, CPU example, and [NIC collector](nic-inventory.md) are implemented.
GPU/NUMA/NIC derivation remains in [PR #29](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/29)
for [#15](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/15).
`from_observations()` adapts exported snapshots without changing the collectors.
Graph tests run without optional dependencies or hardware; integration tests
exercise the NIC collector using sysfs fixtures. No physical affinity validation
is claimed.

## API and stable identities

The public API imports from `topology_scheduler` without Ray, Kubernetes, or
NVML. [topology.py](../topology_scheduler/topology.py) defines `TopologyVertex`,
`TopologyRelationship`, `TopologyGraph`, and the read-only `RELATIONSHIP_TYPES`
mapping of direction, unit, and meaning.

`TopologyVertex(kind, node_id, key, attributes={})` derives IDs from these keys.
Components are percent-encoded so delimiters inside them cannot cause collisions.

| Kind | Collector-supplied key | ID |
| --- | --- | --- |
| `node` | Same as `node_id` | `node:<node_id>` |
| `gpu` | GPU UUID | `node:<node_id>/gpu:<uuid>` |
| `nic` | Stable node-local interface identity including port | `node:<node_id>/nic:<key>` |
| `numa` | Canonical nonnegative decimal NUMA number | `node:<node_id>/numa:<number>` |

For NICs, a collector can use a normalized PCI function plus physical port,
such as `pci:0000:02:00.0/port:1`. Multiple ports need distinct keys. Virtual or
unidentified interfaces need a stable source identity documented by their
collector; the graph never invents array-index keys. Store interface name,
MAC/PCI address, advertised speed with units, and diagnostics in `attributes`.
Unavailable values remain `null`; NUMA `-1` must not create a fictitious domain.

IDs are stable for unchanged node identity and device keys, even if a GPU's
local index changes. The legacy adapter uses Ray node IDs, so identity is
scoped to a Ray node's lifetime, not guaranteed across restarts or hardware
replacement. Every device/domain needs an owning node vertex. Duplicate IDs,
dangling endpoints, and cross-node edges of these intra-node types are rejected.

## Relationship schema

Every edge has `kind`, `source`, `target`, `value`, `state`, `discovery_source`,
`confidence`, `evidence`, and `reason`. Serialized edges also contain
`directed`, `unit`, and `meaning`, checked against `RELATIONSHIP_TYPES` on load.

| Kind | Endpoints / direction | Known value / unit |
| --- | --- | --- |
| `contains` | Node to GPU, NIC, or NUMA; directed | `null`; no unit, structural membership |
| `nvlink` | GPU pair; undirected | Nonnegative integer; `links`, not bandwidth |
| `pcie_ancestry` | Pair of GPU/NIC devices; undirected | Source's nonempty ancestry label; `category` |
| `numa_locality` | GPU or NIC to NUMA domain; directed | `local`; `category` |
| `gpu_nic_affinity` | GPU and NIC; undirected | Proximity label below; `category` |

`evidence` retains JSON values, nested raw records, source paths, endpoint
identities, and diagnostics supplied by the collector. `discovery_source` names
the observation/derivation method and must not be empty. Confidence is `high`,
`medium`, `low`, or `unknown`: a source assessment, not a graph-assigned probability.

State is `known`, `unknown`, or `unsupported`. The last two require `value=null`
and a nonempty `reason`; these edges remain present. Zero known NVLinks differs
from an unavailable count. A missing edge means no record was supplied, not
zero distance or confirmed disconnection. Unknown NUMA information can stay in
device diagnostics or an unknown affinity edge without inventing a NUMA endpoint.
Future unrecognized kinds/schema versions are rejected rather than dropped;
use a supported kind with unsupported state when its observation is unavailable.

## Affinity and node queries

`nics_near_gpu(gpu_id)` and `gpus_near_nic(nic_id)` return every tie for the
best **known** affinity category, sorted by stable vertex ID. `AFFINITY_ORDER`
exposes the ordinal convention:

```text
same-device < same-switch < same-root-complex < same-numa < cross-numa
```

These are categories, not bandwidth, latency, or scheduling weights. Unknown
and unsupported edges do not participate; no known candidate returns an empty
tuple. Cross-NUMA may be the best known candidate when closer relationships
are unknown. Inspect those edges before interpreting a result as physically
closest. Confidence and NIC state remain available to callers; queries do not
select a usable interface, reserve a device, or enforce placement.

`relationships_within_node(node_id)` includes every supplied intra-node edge,
including unknown/unsupported ones. `vertex(id)` retrieves an entity. Missing
IDs raise `KeyError`; querying a NIC as a GPU (or vice versa) raises `ValueError`.
Queries never infer affinity from legacy GPU-pair ancestry or advertised speed.

## Serialization and legacy loading

`as_dict()` and `to_json()` emit schema version `1`, sorted `vertices`, and
sorted `relationships`. JSON keys and undirected endpoint IDs are canonical;
edges sort by kind/source/target. Nested evidence arrays keep their original
order and values. Non-JSON values, non-string keys, NaN, and infinity are
rejected. Duplicate edges require explicit evidence reconciliation by the
caller. Attributes/evidence are copied on input and dictionary export; treat
the stored payloads as snapshots.

`from_dict()` / `from_json()` accept that format or one legacy GPU-only
inventory dictionary (or a list of them). `from_inventory()` takes one existing
`RayNodeInventory`:

```python
from topology_scheduler import TopologyGraph

# inventory is a RayNodeInventory from discover_ray_gpu_inventory().
graph = TopologyGraph.from_inventory(inventory)
assert TopologyGraph.from_json(graph.to_json()).to_json() == graph.to_json()
```

Node metadata and GPU properties become attributes; each original connection
record is retained as evidence on separate NVLink and ancestry edges. Because
legacy records do not provide confidence or unavailable-versus-unsupported
diagnostics, the adapter uses `legacy-inventory`, unknown confidence, and
unknown state for missing/null fields. An `unknown-<value>` NVML category stays
in evidence with unknown state. Device-only records without `connections` load
without fabricated GPU-pair edges. Existing inventory models, discovery output,
and planner conversion remain unchanged. This is a one-way format upgrade:
new graph round trips preserve the converted representation, not legacy array
order or format.

## Example, validation, and enforcement boundary

### Adapting collector snapshots

```python
graph = TopologyGraph.from_observations(
    gpu_inventory,
    nic_inventory=nic_inventory,  # optional NodeNICInventory.as_dict() or object
    host_topology=host_topology,  # optional HostTopology.as_dict() or object
)
```

The GPU argument is one `RayNodeInventory` or its JSON dictionary. Optional
snapshots use the public dictionary shapes from the merged [NIC collector](nic-inventory.md)
and [locality PR #29](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/pull/29).
The adapter only reads these records; it never starts Ray or discovers devices.
NIC collection is available through `discover_nic_inventory()`; host locality
collection remains in PR #29. Callers join collector records by `node_id`
before passing one node here.
Collectors should retain these exported fields, or coordinate a schema update:

| Input | Fields used to construct relationships |
| --- | --- |
| GPU inventory | `node_id`, `devices[].uuid`, `devices[].pci_bus_id`, legacy `connections` |
| NIC inventory | `node_id`, `interfaces[].name`, `pci_address` and `numa_node` Reading objects with `value`/`confidence` |
| Host topology | `node_id`, `nics[].name/pci_address/numa_node`, `gpus[].uuid/pci_address/numa_node/nics` |
| GPU's NIC proximity | `nic_name`, `proximity`, `shared_pci_ancestor`, optional `reason` |

NICs join by interface name within the same Ray node and get keys
`interface:<name>`. Two interfaces on the same PCI function remain distinct;
renaming an interface changes its graph identity. GPU UUIDs remain the keys,
and PCI comparison handles NVML's eight-digit versus sysfs's four-digit domain.
Mismatched node IDs, conflicting known PCI/NUMA identities, duplicate source
records, and dangling GPU/NIC references raise errors rather than silently
combining inconsistent snapshots. Matching identities do not prove samples were
taken simultaneously; callers own freshness and collection coordination.

Original NIC readings (including confidence, source, RDMA data, unknown fields,
and advertised speeds) remain in vertex attributes. Host metadata and
diagnostics remain on the node; GPU/NIC locality records remain on devices,
and per-pair proximity records remain in edge evidence. Record arrays are
indexed by their stable identities; evidence arrays such as PCI paths retain
their meaning and order. Unknown NUMA values stay in attributes without a
fabricated NUMA vertex. No NUMA membership is inferred from an affinity label.
Only the collector's proximity label determines affinity; the adapter never
recomputes it from speed or ancestry. Missing pairs get explicit unknown edges,
and unsupported pairs stay unsupported. An observed shared ancestor creates
a separate `pcie_ancestry` edge with value `shared-ancestor` and the raw ancestor
in evidence. Edge confidence stays `unknown` because these snapshots do not
provide an edge-confidence assessment; per-field confidence remains intact.

To process exported per-node JSON files without running discovery:

```bash
python -m examples.topology_graph --gpu-inventory gpu.json --nic-inventory nics.json --host-topology host.json
```

Supply each collector's single-node dictionary, not an example command's output
wrapper or a whole-cluster array. The file mode labels its input source and
makes no claim that supplied files came from physical hardware. Omitting NIC
or locality input preserves the available information; GPU-only behavior is
unchanged. The adapter tests cover these source schemas, conflicts, unknowns,
equal-distance ties, and canonical round trips.

The adapter suite also runs the merged NIC collector against in-memory sysfs
fixtures, then converts its actual `NodeNICInventory` objects and JSON exports.
It checks attached and unmatched RDMA evidence, advertised speed, diagnostics,
unreadable/unsupported fields, NUMA membership, and affinity ties when a locality
fixture is supplied. This is collector integration coverage without physical
hardware; it does not validate live GPU-to-NIC proximity.

From the repository root, run:

```bash
python -m examples.topology_graph
```

The [example](../examples/topology_graph.py) labels its output `synthetic_inputs`,
prints a two-GPU/three-NIC/two-NUMA graph, and shows ties, reverse queries, and
an unsupported relationship. It needs no optional dependencies or hardware.
[Tests](../tests/test_topology.py) cover legacy loading, missing fields, raw
evidence, canonical serialization, multiple nodes/domains/NICs, and ties.

Observed topology, derived affinity, planner preference, and backend enforcement
remain separate. Neither the policy nor Ray/KAI binds workers to GPU/NIC IDs
from this graph. No policy weight, reservation, or runtime dependency changes.
Active measurements belong to
[#17](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/17);
future measured edges need direction, units, timestamps, and provenance without
replacing the physical observations represented here.
