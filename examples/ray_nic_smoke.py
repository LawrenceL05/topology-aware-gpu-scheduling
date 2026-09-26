"""Two local Ray nodes read this host's real NICs; no GPU or network benchmark."""

import json

import ray
from ray.cluster_utils import Cluster

from topology_scheduler import discover_nic_inventory


def main():
    cluster = Cluster()
    try:
        for name in ("a", "b"):
            cluster.add_node(num_cpus=1, num_gpus=0, include_dashboard=False,
                             resources={f"topology_node:{name}": 1})
        ray.init(address=cluster.address, log_to_driver=False)
        expected = {
            name: next(node["NodeID"] for node in ray.nodes()
                       if node["Alive"] and f"topology_node:{name}" in node["Resources"])
            for name in ("a", "b")
        }
        records = discover_nic_inventory()
        assert [(r.node_name, r.node_id) for r in records] == sorted(expected.items())
        assert len({r.node_id for r in records}) == 2
        for record in records:
            loopback = next(interface for interface in record.interfaces if interface.name == "lo")
            assert loopback.kind == "loopback"
        repeated = discover_nic_inventory()
        assert [r.as_dict() for r in records] == [r.as_dict() for r in repeated]
        print(json.dumps({"same_host": True, "benchmark_evidence": False,
                          "nodes": [record.as_dict() for record in records]}, indent=2))
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
