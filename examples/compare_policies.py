"""Print synthetic policy decisions as JSON; this is not benchmark evidence."""

import json

from topology_scheduler import Node, PolicyName, Workload, plan_with_record


nodes = [
    Node("a", "H100", 2, 80),
    Node("b", "B200", 2, 180),
    Node("c", "H100", 2, 80),
]
workload = Workload(2, 40, {"H100": 10, "B200": 7}, 20)
bandwidth = {("a", "b"): 10, ("a", "c"): 100, ("b", "c"): 10}

records = []
for policy in PolicyName:
    options = {"accelerator_type": "H100"} if policy is PolicyName.ACCELERATOR_TYPE else {}
    _, record = plan_with_record(
        nodes, workload, bandwidth, policy=policy, **options
    )
    records.append(record.as_dict())

print(json.dumps({"synthetic_inputs": True, "records": records}, indent=2))
