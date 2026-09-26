"""Adapt collector snapshots to the graph; never collect or infer proximity."""

from dataclasses import asdict, is_dataclass, replace

from .topology import TopologyGraph, TopologyRelationship, TopologyVertex, _json_copy, _text


def _snapshot(value, label):
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    elif is_dataclass(value):
        value = asdict(value)
        # Existing GPU inventory dataclasses contain tuples; their JSON shape
        # is arrays. Collector as_dict() methods already export plain JSON.
        def arrays(item):
            if isinstance(item, (tuple, list)):
                return [arrays(child) for child in item]
            if isinstance(item, dict):
                return {key: arrays(child) for key, child in item.items()}
            return item
        value = arrays(value)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a snapshot object")
    return _json_copy(value)


def _indexed(records, key, label):
    if not isinstance(records, list):
        raise ValueError(f"{label} must be an array")
    result = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{label} entries must be objects")
        identity = record.get(key)
        _text(identity, f"{label}.{key}")
        if identity in result:
            raise ValueError(f"Duplicate {label} identity: {identity}")
        result[identity] = record
    return result


def _reported(record, field):
    reading = record.get(field)
    if reading is None:
        return None
    if not isinstance(reading, dict) or "confidence" not in reading:
        raise ValueError(f"NIC {field} must be a Reading object")
    return reading.get("value") if reading["confidence"] == "reported" else None


def _numa(value):
    if value is None or type(value) is int and value == -1:
        return None
    if type(value) is not int or value < 0:
        raise ValueError("NUMA identity must be a nonnegative integer, -1, or null")
    return value


def _same_pci(left, right):
    # Compare NVML's eight-digit domain with sysfs's four-digit form, without
    # changing either source record or treating missing PCI as a match.
    def canonical(value):
        domain, bus, function = value.lower().split(":")
        return int(domain, 16), bus, function
    return canonical(left) == canonical(right)


def graph_from_observations(gpu_inventory, nic_inventory, host_topology):
    gpu = _snapshot(gpu_inventory, "gpu_inventory")
    base = TopologyGraph.from_legacy(gpu)
    if nic_inventory is None and host_topology is None:
        return base
    node_id = gpu["node_id"]
    nic = _snapshot(nic_inventory, "nic_inventory") if nic_inventory is not None else None
    host = _snapshot(host_topology, "host_topology") if host_topology is not None else None
    for label, snapshot in (("nic_inventory", nic), ("host_topology", host)):
        if snapshot is not None and snapshot.get("node_id") != node_id:
            raise ValueError(f"{label} node_id does not match GPU inventory")

    interfaces = _indexed(nic["interfaces"], "name", "NIC interfaces") if nic else {}
    host_nics = _indexed(host["nics"], "name", "host NICs") if host else {}
    host_gpus = _indexed(host["gpus"], "uuid", "host GPUs") if host else {}
    gpu_vertices = {v.key: v for v in base.vertices if v.kind == "gpu"}
    if host_gpus.keys() - gpu_vertices.keys():
        raise ValueError("Host topology references GPU UUIDs absent from GPU inventory")
    vertices = {v.id: v for v in base.vertices}
    edges = list(base.relationships)
    node = next(v for v in base.vertices if v.kind == "node")
    metadata = dict(node.attributes)
    if nic:
        metadata["nic_inventory"] = {k: v for k, v in nic.items() if k != "interfaces"}
    if host:
        metadata["host_topology"] = {k: v for k, v in host.items() if k not in {"gpus", "nics"}}
    vertices[node.id] = replace(node, attributes=metadata)

    def add_vertex(vertex, source):
        if vertex.id not in vertices:
            vertices[vertex.id] = vertex
            edges.append(TopologyRelationship("contains", node.id, vertex.id, source))

    def locality(vertex, domain, source, evidence):
        domain = _numa(domain)
        if domain is None:
            return
        numa = TopologyVertex("numa", node_id, str(domain))
        add_vertex(numa, source)
        edges.append(TopologyRelationship("numa_locality", vertex.id, numa.id,
                                          source, value="local", evidence=evidence))

    nic_vertices = {}
    for name in sorted(interfaces.keys() | host_nics.keys()):
        observed, located = interfaces.get(name), host_nics.get(name)
        attrs = {"name": name, "identity_scope": "node-local interface name"}
        observed_numa = _numa(_reported(observed, "numa_node")) if observed else None
        located_numa = _numa(located.get("numa_node")) if located else None
        if observed and located:
            observed_pci, located_pci = _reported(observed, "pci_address"), located.get("pci_address")
            if observed_pci and located_pci and not _same_pci(observed_pci, located_pci):
                raise ValueError(f"Conflicting PCI identity for NIC {name}")
            if observed_numa is not None and located_numa is not None and observed_numa != located_numa:
                raise ValueError(f"Conflicting NUMA identity for NIC {name}")
        if observed:
            attrs["nic_inventory"] = observed
        if located:
            attrs["host_topology"] = located
        vertex = TopologyVertex("nic", node_id, f"interface:{name}", attrs)
        nic_vertices[name] = vertex
        source = "nic-inventory" if observed else "host-topology"
        add_vertex(vertex, source)
        if observed_numa is not None:
            locality(vertex, observed_numa, "nic-inventory", {"numa_node": observed["numa_node"]})
        elif located_numa is not None:
            locality(vertex, located_numa, "host-topology", {"nic": located, "sources": host.get("sources", {})})

    for uuid, vertex in sorted(gpu_vertices.items()):
        located = host_gpus.get(uuid)
        proximities = {}
        if located:
            left, right = vertex.attributes.get("pci_bus_id"), located.get("pci_address")
            if left and right and not _same_pci(left, right):
                raise ValueError(f"Conflicting PCI identity for GPU {uuid}")
            attrs = dict(vertex.attributes)
            # Store per-NIC records on edges, so input ordering cannot change
            # canonical JSON. Preserve every remaining locality field here.
            attrs["host_topology"] = {k: v for k, v in located.items() if k != "nics"}
            vertices[vertex.id] = replace(vertex, attributes=attrs)
            locality(vertex, located.get("numa_node"), "host-topology",
                     {"gpu": attrs["host_topology"], "sources": host.get("sources", {})})
            proximities = _indexed(located["nics"], "nic_name", "GPU NIC proximity")
            if proximities.keys() - nic_vertices.keys():
                raise ValueError(f"Affinity for GPU {uuid} references an unknown NIC")
        for name, peer in sorted(nic_vertices.items()):
            proximity = proximities.get(name)
            evidence = {"sources": host.get("sources", {}) if host else {}}
            category = None
            reason = "No host-topology proximity observation for this GPU/NIC pair"
            if proximity:
                evidence["observation"] = proximity
                category = proximity.get("proximity")
                if category in (None, "unknown", "unsupported"):
                    reason = proximity.get("reason") or "Collector reports unknown proximity"
                    category = None
                else:
                    reason = None
            state = ("known" if category else "unsupported" if proximity and
                     proximity.get("proximity") == "unsupported" else "unknown")
            edges.append(TopologyRelationship(
                "gpu_nic_affinity", vertex.id, peer.id, "host-topology",
                value=category, state=state, evidence=evidence, reason=reason,
            ))
            if proximity:
                shared = proximity.get("shared_pci_ancestor")
                if shared is not None:
                    _text(shared, "shared_pci_ancestor")
                edges.append(TopologyRelationship(
                    "pcie_ancestry", vertex.id, peer.id, "host-topology",
                    value="shared-ancestor" if shared else None,
                    state="known" if shared else "unknown", evidence=evidence,
                    reason=None if shared else "No shared PCI ancestor reported by collector",
                ))
    return TopologyGraph(tuple(vertices.values()), tuple(edges))
