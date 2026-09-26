# GPU, NUMA, and NIC host locality

See [current implementation, validation status, and versions](current-status.md)
for the shared support summary and evidence boundaries.

[V1.2 discovery](v1.2-topology-discovery.md) records how GPUs relate to each
other inside a node. This collector answers the host-side question next to it:
which CPU NUMA domain owns each GPU, and which network interface sits closest
to it. [host_topology.py](../topology_scheduler/host_topology.py) reads that
from the kernel, so no one types a GPU, NUMA, NIC, or PCI mapping by hand.

Everything here is an observation. Proximity does not promise bandwidth, and
nothing in this module binds a GPU or a NIC to a workload.

```mermaid
flowchart LR
    N[NVML: GPU UUID and PCI bus ID] --> C[collect_host_topology]
    I["NIC inventory: interfaces,<br/>PCI function, NUMA, speed"] --> C
    S["sysfs: PCI tree and<br/>numa_node per function"] --> C
    C --> G[GPULocality: NUMA node and PCI path]
    C --> P[NICProximity per interface, with evidence]
    C --> D[Diagnostics for every unknown]
    G --> R[Stable, sorted serialization]
    P --> R
```

## Where each half comes from

The interfaces are **not** read here. [`collect_nic_inventory()`](nic-inventory.md)
already reads every interface under `/sys/class/net` with each field's source
and confidence, so this collector consumes that record and adds only what
proximity needs and an inventory cannot carry: where each PCI function sits in
the host's device tree.

| Source | Read by | Used for |
| --- | --- | --- |
| `/sys/class/net/*` | NIC inventory | Interface identity, kind, MAC, MTU, driver, state, advertised speed, RDMA |
| `/sys/class/net/<name>/device/uevent` | NIC inventory | The interface's PCI function |
| `/sys/class/net/<name>/device/numa_node` | NIC inventory | The NUMA node of an interface |
| `/sys/devices/pci*/**` | This module | The PCI hierarchy, walked by directory name so each function's ancestry is known |
| `<gpu pci function>/numa_node` | This module | The NUMA node of a GPU |

GPU identity still comes from NVML through the existing V1.2 probe: the UUID
and PCI bus ID are the stable identifiers, and NVML's eight-digit domain is
normalized to the four-digit sysfs form by `normalize_pci_address()`.

Both collectors share one `SysfsReader`, the inventory's, which is the only
part that touches a filesystem. Fixtures subclass it, so the classification
logic is tested without a host, a GPU, or root access. Passing `inventory=` to
`collect_host_topology()` reuses an inventory already read from the same host
instead of collecting a second one.

Each `HostNIC` keeps its inventory record on `nic.interface`, so per-field
provenance survives into the map: `nic.interface.speed_mbps` still names the
file it came from and how much the answer is worth, while `nic.speed_mbps`,
`nic.numa_node`, `nic.kind`, and `nic.operstate` are the plain values.

## How proximity is decided

For each GPU and interface, the first rule that matches wins:

| Proximity | Rule |
| --- | --- |
| `same-device` | The two functions share a PCI device, differing only in function number |
| `same-switch` | Their PCI ancestors share at least one bridge below the host bridge |
| `same-root-complex` | They share only the host bridge |
| `same-numa` | No shared PCI ancestor, and both NUMA nodes are known and equal |
| `cross-numa` | No shared PCI ancestor, and both NUMA nodes are known and differ |
| `unknown` | Anything else, always with a reason |

Each result keeps the evidence behind it: the deepest shared PCI ancestor, both
NUMA nodes, and a reason when the answer is `unknown`. A GPU's `nearest_nic` is
the first interface in that order, or `None` when nothing could be related.

## Unknown states and fallbacks

Nothing is guessed, and nothing is dropped. Every interface stays in the
result, and every gap adds a diagnostic naming the GPU or interface:

| Situation | Result |
| --- | --- |
| `numa_node` holds `-1` | NUMA node is `None`; the kernel does not know |
| `numa_node` is absent | NUMA node is `None`, reported as `unavailable` |
| `numa_node` cannot be read | NUMA node is `None`, reported as `unreadable` |
| The GPU's PCI address is not under `/sys/devices` | The GPU is kept with an empty PCI path and every interface is `unknown` |
| An interface has no PCI function | It is kept with the inventory's kind, `loopback` or `virtual`, and classified `unknown` |
| An interface's PCI function is not under `/sys/devices` | It is kept with an empty PCI path, a diagnostic, and `unknown` |
| `speed` is `-1` or unreadable | `speed_mbps` is `None`, and the inventory records which of the two it was |
| No interfaces exist at all | The inventory's own diagnostic is carried through, and the GPU list is still returned |

Only the two findings that change a proximity answer — an unusable PCI address
and an unknown NUMA node — are repeated as diagnostics here. Every other
per-field problem stays where the inventory recorded it, on `nic.interface`,
rather than being copied into a second list that could drift from the first.

## Stable identity and ordering

Repeated discovery on an unchanged host produces identical output: GPUs are
sorted by PCI address, interfaces by name, and each GPU's interfaces by
proximity and then name. `as_dict()` contains only plain types, so a report can
be compared or stored directly.

## Run it

```bash
python -m examples.host_topology
python -m examples.host_topology --host
python -m examples.host_topology --live
```

The default prints a synthetic two-socket host, so the output is deterministic
and needs no GPU; it is the version CI runs. `--host` reads this machine's
sysfs and supplies no GPUs, which is a quick way to see the interfaces a real
host reports. `--live` connects to a running Ray cluster and maps every GPU
node through `discover_host_topology()`, which pins one probe per node:

```python
import ray
from topology_scheduler import discover_host_topology

ray.init(address="auto")
for topology in discover_host_topology():
    for gpu in topology.gpus:
        print(topology.node_name, gpu.uuid, gpu.numa_node,
              gpu.nearest_nic.nic_name if gpu.nearest_nic else None)
```

## Confidence limits

- Proximity is structural. A NIC on the same switch is not necessarily faster
  in practice; only a measurement says that, and none is implied here.
- Nothing binds a GPU to a NIC. Ray, KAI, and the driver still choose devices,
  and this module does not change placement or its cost function.
- sysfs is Linux-only. On other platforms the collector returns no PCI devices
  and no interfaces, with diagnostics rather than errors.
- Virtualized and containerized hosts often hide the PCI tree or report no NUMA
  node. Those cases stay `unknown`; they are not approximated.
- Advertised speed comes from the inventory and is what the driver reports the
  link negotiated, never application throughput.
- These observations are not yet a typed graph;
  [affinity graph #16](https://github.com/LawrenceL05/topology-aware-gpu-scheduling/issues/16)
  is where placement policies would consume them.
- No physical multi-socket GPU host has been mapped yet. The rules are covered
  by fixtures, and CI additionally runs the real reader against a Linux host
  that has interfaces but no GPU.
