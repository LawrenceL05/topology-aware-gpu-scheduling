ARG DYNAMO_BASE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2@sha256:b23ce9e87c413725024ddfd2e06c1e7da220eb48f8a93037f0b01d564b54bd13
FROM ${DYNAMO_BASE}

USER root
COPY deploy/dynamo-v1/requirements.txt /tmp/topology-scheduler-requirements.txt
RUN python3 -m pip install --no-cache-dir \
    --requirement /tmp/topology-scheduler-requirements.txt

WORKDIR /opt/topology-aware-gpu-scheduling
COPY pyproject.toml README.md ./
COPY topology_scheduler ./topology_scheduler
RUN python3 -m pip install --no-cache-dir --no-deps .

USER dynamo
