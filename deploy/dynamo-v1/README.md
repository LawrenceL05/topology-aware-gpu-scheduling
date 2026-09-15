# V1 Dynamo environment recipe

This directory records the exact environment selected by the
[V1 Dynamo contract](../../docs/dynamo-v1-contract.md). It prepares an image for
the worker-lifecycle implementation tracked in issue 2; it does not claim that
the integration or a real-GPU run already exists.

Build from the repository root on a Linux/amd64 Docker host:

```bash
docker build \
  --file deploy/dynamo-v1/Dockerfile \
  --tag topology-scheduler-dynamo:v1 \
  .
```

The base reference includes the multi-architecture image digest. The contract
also records the resolved Linux/amd64 platform digest. Verify both before a
test run:

```bash
docker buildx imagetools inspect \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2
```

The derived image installs Ray 2.55.0 exactly and installs this project without
changing Dynamo's pinned backend dependencies. Use this same image for Ray
head and worker nodes so their Python and Ray versions match.

Before a run, validate the host and image:

```bash
nvidia-smi
docker run --rm topology-scheduler-dynamo:v1 \
  python3 -c 'import ray, vllm; print(ray.__version__, vllm.__version__)'
```

The expected output contains `2.55.0 0.26.0`. The image requires an NVIDIA
580.00.03 or newer driver. Model downloads use the exact revision in
`contract.json`; pass credentials through the runtime environment when a model
requires them. Do not store tokens in this repository or image.
