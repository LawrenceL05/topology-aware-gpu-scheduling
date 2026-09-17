"""Print a synthetic typed topology graph and categorical affinity queries."""

import json

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
    graph = example_graph()
    gpu = next(vertex for vertex in graph.vertices if vertex.kind == "gpu")
    nic = next(vertex for vertex in graph.vertices if vertex.kind == "nic")
    print(json.dumps({
        "synthetic_inputs": True,
        "graph": graph.as_dict(),
        "queries": {
            "gpu": gpu.id,
            "nics_near_gpu": [vertex.id for vertex in graph.nics_near_gpu(gpu.id)],
            "nic": nic.id,
            "gpus_near_nic": [vertex.id for vertex in graph.gpus_near_nic(nic.id)],
            "intra_node_relationship_count": len(graph.relationships_within_node(gpu.node_id)),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
