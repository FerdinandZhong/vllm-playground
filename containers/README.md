# Container Definitions

This directory contains various Containerfile definitions for different deployment scenarios.

## Available Containerfiles

### 🚀 Containerfile.vllm (Recommended for GPU)
- **Base Image**: Official vLLM OpenAI-compatible server image
- **Best For**: GPU deployments, production environments
- **Features**: Pre-built vLLM, smaller image size, faster builds
- **Platform**: Linux x86_64 with CUDA

```bash
podman build -f containers/Containerfile.vllm -t vllm-playground:vllm .
```

### 🔥 Containerfile.cuda
- **Base Image**: Red Hat Universal Base Image 9 (UBI9) Minimal
- **Best For**: RHEL/CentOS environments, GPU deployments
- **Features**: Full control, builds vLLM from PyPI with CUDA support
- **Platform**: RHEL 9 x86_64 with CUDA

```bash
podman build -f containers/Containerfile.cuda -t vllm-playground:cuda .
```

### 🍎 Containerfile.mac
- **Base Image**: Python 3.11 slim
- **Best For**: macOS development, CPU-only inference
- **Features**: Optimized for Apple Silicon, CPU mode
- **Platform**: macOS (ARM64/x86_64)

```bash
podman build -f containers/Containerfile.mac -t vllm-playground:mac .
```

### 🏢 Containerfile.rhel9
- **Base Image**: Red Hat Universal Base Image 9 (UBI9) Minimal
- **Best For**: Enterprise RHEL deployments
- **Features**: Based on official RHEL images, enterprise support
- **Platform**: RHEL 9 x86_64

```bash
podman build -f containers/Containerfile.rhel9 -t vllm-playground:rhel9 .
```

## Quick Start

### GPU Deployment (Recommended)
```bash
# Using official vLLM image
podman build -f containers/Containerfile.vllm -t vllm-playground:latest .
podman run -d --name vllm-playground \
  --device nvidia.com/gpu=all \
  -p 7860:7860 -p 8000:8000 \
  vllm-playground:latest
```

### CPU Deployment (macOS)
```bash
podman build -f containers/Containerfile.mac -t vllm-playground:mac .
podman run -d --name vllm-playground \
  -p 7860:7860 -p 8000:8000 \
  vllm-playground:mac
```

## Image Comparison

| Containerfile | Base Image | Size | Build Time | Best For |
|--------------|------------|------|------------|----------|
| vllm | vLLM official | ~5GB | Fast | GPU production |
| cuda | UBI9 minimal | ~6GB | Medium | RHEL/GPU custom |
| mac | Python slim | ~3GB | Fast | macOS/CPU dev |
| rhel9 | UBI9 minimal | ~6GB | Medium | Enterprise RHEL |

## Port Mappings

All containers expose the same ports:
- **7860**: WebUI interface
- **8000**: vLLM API server

## Environment Variables

Common environment variables across all containers:
- `WEBUI_PORT`: WebUI port (default: 7860)
- `VLLM_PORT`: vLLM API port (default: 8000)
- `CUDA_VISIBLE_DEVICES`: GPU devices (GPU containers only)

For deployment examples, see the `../deployments/` directory.

