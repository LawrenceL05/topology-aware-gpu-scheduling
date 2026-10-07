"""Print a synthetic graph, or adapt one node's exported collector snapshots."""

import argparse
import json
from pathlib import Path

from topology_scheduler import (
    GPUConnection, GPUDevice, RayNodeInventory, TopologyGraph,
    TopologyRelationship, TopologyVertex,
)


def example_graph():
    inventory = RayNodeInventory(
        "example-node", "a", "topology_node:a", 2,
        (GPUDevice(0, "GPU-a", "NVIDIA H100", "H100", 80, "0000:01:00.0"),
         GPUDevice(1, "GPU-b", "NVIDIA H100", "H100", 80, "0000:81:00.0")),
        (GPUConnection("GPU-a", "GPU-b", "system", 0),),
    )
    legacy = TopologyGraph.from_inventory(inventory)
    node_id = inventory.node_id
    node = TopologyVertex("node", node_id, node_id)
    gpu_a = TopologyVertex("gpu", node_id, "GPU-a")
    gpu_b = TopologyVertex("gpu", node_id, "GPU-b")
    nics = [TopologyVertex("nic", node_id, f"pci:{pci}/port:1", {
        "name": name, "pci_bus_id": pci, "advertised_speed_gbps": 100,
        "discovery_source": "synthetic-fixture",
    }) for name, pci in (("eth0", "0000:02:00.0"), ("eth1", "0000:03:00.0"),
                        ("eth2", "0000:82:00.0"))]
    domains = [TopologyVertex("numa", node_id, str(i)) for i in range(2)]
    vertices = legacy.vertices + tuple(nics + domains)
    edges = list(legacy.relationships)
    for vertex in nics + domains:
        edges.append(TopologyRelationship("contains", node.id, vertex.id, "synthetic-fixture"))
    for vertex, domain in ((gpu_a, 0), (gpu_b, 1), (nics[0], 0), (nics[1], 0), (nics[2], 1)):
        edges.append(TopologyRelationship(
            "numa_locality", vertex.id, domains[domain].id, "synthetic-fixture",
            value="local", evidence={"numa_node": domain}))
    for gpu, categories in ((gpu_a, ("same-numa", "same-numa", "cross-numa")),
                            (gpu_b, ("cross-numa", None, "same-numa"))):
        for nic, category in zip(nics, categories):
            edges.append(TopologyRelationship(
                "gpu_nic_affinity", gpu.id, nic.id, "synthetic-fixture",
                value=category, state="known" if category else "unsupported",
                confidence="medium" if category else "unknown",
                evidence={"gpu_uuid": gpu.key, "nic_pci": nic.attributes["pci_bus_id"],
                          "method": "illustrative NUMA classification, no discovery"},
                reason=None if category else "Fixture simulates inaccessible PCI ancestry",
            ))
    return TopologyGraph(vertices, tuple(edges))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-inventory", type=Path, help="one GPU inventory JSON object")
    parser.add_argument("--nic-inventory", type=Path, help="one NodeNICInventory.as_dict() JSON object")
    parser.add_argument("--host-topology", type=Path, help="one HostTopology.as_dict() JSON object")
    args = parser.parse_args()
    if (args.nic_inventory or args.host_topology) and not args.gpu_inventory:
        parser.error("--nic-inventory and --host-topology require --gpu-inventory")
    def read(path):
        return json.loads(path.read_text(encoding="utf-8")) if path else None
    graph = (TopologyGraph.from_observations(
        read(args.gpu_inventory), nic_inventory=read(args.nic_inventory),
        host_topology=read(args.host_topology),
    ) if args.gpu_inventory else example_graph())
    gpu = next((vertex for vertex in graph.vertices if vertex.kind == "gpu"), None)
    nic = next((vertex for vertex in graph.vertices if vertex.kind == "nic"), None)
    print(json.dumps({
        **({"synthetic_inputs": True} if not args.gpu_inventory else {}),
        "input_source": "snapshot_files" if args.gpu_inventory else "synthetic_fixture",
        "benchmark_evidence": False,
        "graph": graph.as_dict(),
        "queries": {
            "gpu": gpu.id if gpu else None,
            "nics_near_gpu": [vertex.id for vertex in graph.nics_near_gpu(gpu.id)] if gpu else [],
            "nic": nic.id if nic else None,
            "gpus_near_nic": [vertex.id for vertex in graph.gpus_near_nic(nic.id)] if nic else [],
            "intra_node_relationship_count": len(graph.relationships),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
