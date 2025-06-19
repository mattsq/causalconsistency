# syntax=docker/dockerfile:1
# Build argument DEVICE chooses the base image (cpu or cuda)
ARG DEVICE=cpu
ARG PYTHON_VERSION=3.10
ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

FROM python:${PYTHON_VERSION}-slim AS cpu
# Install base packages for CPU image
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

FROM ${CUDA_IMAGE} AS cuda
# Install Python on the CUDA image
RUN apt-get update && apt-get install -y python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/local/bin/python \
    && ln -s /usr/bin/pip3 /usr/local/bin/pip

FROM ${DEVICE} AS final
WORKDIR /app
COPY pyproject.toml README.md LICENSE conda-lock.yml ./
RUN pip install --no-cache-dir pip setuptools wheel
RUN pip install --no-cache-dir .
COPY src src
COPY examples examples
ENTRYPOINT ["python", "-m", "causal_consistency_nn"]
CMD ["--help"]
