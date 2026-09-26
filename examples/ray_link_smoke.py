"""Two local Ray nodes on one host; tests link-probe orchestration only."""
import json
import sys

import ray
from ray.cluster_utils import Cluster

from topology_scheduler import (
    Node, ProbeParameters, Workload, measure_ray_links, plan_with_record,
    resolve_link_costs,
)


def main():
    # Cluster is a Ray testing utility. Both nodes share this host, so the probe
    # traffic never leaves it and the numbers say nothing about a real network.
    cluster = Cluster()
    try:
        for name in ("a", "b"):
            cluster.add_node(num_cpus=1, include_dashboard=False,
                             resources={f"topology_node:{name}": 1})
        ray.init(address=cluster.address)
        report = measure_ray_links(
            parameters=ProbeParameters(duration_seconds=0.5, warmup_seconds=0.1,
                                       latency_samples=5, timeout_seconds=20),
            allow_network_load=True,  # Host-local traffic between two local nodes.
        )
        assert [item.direction for item in report.measurements] == ["a->b", "b->a"]
        assert all(item.status == "succeeded" for item in report.measurements), \
            report.as_dict()
        for item in report.measurements:
            for endpoint in (item.source, item.destination):
                # This smoke's Linux nodes use a host primary IPv4 address.
                # Exercise real sysfs evidence on both the client and actor.
                if sys.platform.startswith("linux"):
                    assert endpoint.nic is not None, endpoint
                    assert endpoint.nic["name"] == endpoint.interface
                    assert endpoint.nic_collected_at is not None
                    assert endpoint.interface_source == "Linux SIOCGIFADDR primary IPv4"
                    assert "confidence" in endpoint.nic["speed_mbps"]
                else:
                    assert endpoint.nic is None and endpoint.diagnostics, endpoint
        resolution = resolve_link_costs(report, ["a", "b"], max_age_seconds=300)
        _, planning = plan_with_record(
            [Node(name, "SIMULATED", 1, 80) for name in ("a", "b")],
            Workload(2, 1, {"SIMULATED": 1}, 1),
            resolution.bandwidth_gbps, policy="combined",
            link_costs=resolution.costs,
        )
        assert planning.inputs["bandwidth_sources"]["a|b"]["source"] == "measured"
        print(json.dumps({"host_local_only": True, "report": report.as_dict(),
                          "planning": planning.as_dict()}, indent=2))
    finally:
        ray.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
