"""Discover real NVIDIA GPUs on every live Ray node."""

from dataclasses import asdict

import ray

from topology_scheduler import discover_ray_gpu_inventory


def main():
    ray.init(address="auto")
    try:
        for node in discover_ray_gpu_inventory():
            print(asdict(node))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
