# Ray Backend Integration with vLLM Playground UI

The Ray backend is now fully integrated with the vLLM Playground web interface! This document explains how to use Ray mode through the UI.

## Quick Start

### 1. Install Ray Support

```bash
pip install vllm-playground[ray]
```

### 2. Start Ray Cluster (Optional)

If you want to use an existing Ray cluster:

```bash
cd vllm_playground/ray_serve
python launch_cluster.py --config configs/local_cpu.yaml start
```

If you don't start a cluster manually, the Ray backend will automatically start a local cluster when needed.

### 3. Launch vLLM Playground UI

```bash
# Using the CLI (recommended)
vllm-playground start --host 0.0.0.0 --port 7860

# Or using Python module directly
python -m vllm_playground

# Or from the package directory
cd vllm_playground
python app.py
```

### 4. Select Ray Mode in UI

1. Open http://localhost:7860 in your browser
2. In the configuration panel, set **Run Mode** to `ray`
3. (Optional) Set **Ray Address** if connecting to specific cluster (e.g., `127.0.0.1:6379`)
4. Configure your model settings
5. Click "Start Server"

The UI will:
- Connect to Ray cluster (or start one automatically)
- Deploy vLLM with Ray Serve
- Show deployment progress in the logs
- Make the model available at the configured port

## Configuration Options

### Run Mode

The UI now supports three run modes:

1. **subprocess** - Run vLLM directly as a subprocess (default)
2. **container** - Run vLLM in a Podman/Docker container
3. **ray** - Run vLLM with Ray Serve (NEW!)

### Ray-Specific Settings

When using Ray mode, you can configure:

- **Ray Address** (optional):
  - Leave empty to auto-detect or start local cluster
  - Set to connect to specific cluster (e.g., `127.0.0.1:6379`)
  - Use for remote clusters (e.g., `192.168.1.100:6379`)

All other vLLM settings (model, tensor parallelism, GPU memory, etc.) work the same way across all run modes.

## Usage Examples

### Example 1: Local Ray Cluster (Auto-Start)

**Configuration in UI**:
```
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Run Mode: ray
Ray Address: (leave empty)
Port: 8000
```

**What happens**:
1. UI detects no Ray cluster running
2. Automatically starts local Ray cluster
3. Deploys model with Ray Serve
4. Model available at http://localhost:8000

### Example 2: Pre-Started Ray Cluster

**Terminal 1** - Start cluster:
```bash
cd vllm_playground/ray_serve
python launch_cluster.py --config configs/local_gpu.yaml start
# Note the address: 127.0.0.1:6379
```

**UI Configuration**:
```
Model: meta-llama/Llama-2-7b-chat-hf
Run Mode: ray
Ray Address: 127.0.0.1:6379
Tensor Parallel Size: 1
Port: 8000
HF Token: your_token_here
```

**What happens**:
1. UI connects to existing Ray cluster
2. Deploys Llama model with Ray Serve
3. Model uses cluster's GPU resources
4. Model available at http://localhost:8000

### Example 3: Multi-GPU Tensor Parallelism

**Terminal 1** - Start cluster with 4 GPUs:
```bash
cd vllm_playground/ray_serve
# Edit configs/local_gpu.yaml: num_gpus: 4
python launch_cluster.py --config configs/local_gpu.yaml start
```

**UI Configuration**:
```
Model: meta-llama/Llama-2-70b-chat-hf
Run Mode: ray
Ray Address: 127.0.0.1:6379
Tensor Parallel Size: 4
GPU Memory Utilization: 0.9
Port: 8000
HF Token: your_token_here
```

**What happens**:
1. UI connects to 4-GPU Ray cluster
2. Deploys 70B model with 4-way tensor parallelism
3. Ray automatically creates placement groups
4. Model spans across all 4 GPUs

### Example 4: Remote Ray Cluster

**On remote server** (192.168.1.100):
```bash
cd vllm_playground/ray_serve
python launch_cluster.py --config configs/local_gpu.yaml start
```

**UI Configuration** (on local machine):
```
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Run Mode: ray
Ray Address: 192.168.1.100:6379
Port: 8000
```

**What happens**:
1. UI connects to remote Ray cluster
2. Deploys model on remote server
3. Model runs on remote GPUs
4. API accessible at http://192.168.1.100:8000

## UI Features with Ray Mode

### Start Server

When you click "Start Server" in Ray mode:

1. **Connection Phase**:
   - UI shows: "Connecting to Ray cluster..."
   - Displays cluster info (mode, address, nodes)

2. **Deployment Phase**:
   - UI shows: "Deploying model with Ray Serve..."
   - Shows progress: "This may take several minutes..."

3. **Ready Phase**:
   - UI shows: "✅ Model deployed successfully"
   - Shows application name and startup time
   - Enables chat interface

### Stop Server

When you click "Stop Server":

1. UI shows: "Stopping Ray Serve deployment..."
2. Stops the vLLM deployment
3. Keeps Ray cluster running (for reuse)
4. UI shows: "✅ Ray deployment stopped"

**Note**: Stopping the deployment does NOT stop the Ray cluster. The cluster continues running for future deployments.

### Server Status

The status indicator shows:
- ✅ Green: Ray deployment running and healthy
- ⚠️ Yellow: Ray deployment starting
- ❌ Red: Ray deployment stopped or error

## API Endpoints

When using Ray mode, all OpenAI-compatible endpoints work normally:

```bash
# Health check
curl http://localhost:8000/health

# Text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Hello!",
    "max_tokens": 50
  }'

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Monitoring

### Ray Dashboard

Access the Ray dashboard at http://localhost:8265 to monitor:
- Resource utilization (CPU, GPU, Memory)
- Active deployments
- Request metrics
- Logs and traces

### vLLM Playground Logs

The UI shows deployment logs in real-time:
- Ray cluster connection status
- Model download progress
- Deployment initialization
- Readiness checks
- Any errors or warnings

## Advantages of Ray Mode

### vs Subprocess Mode

✅ **Better resource management** - Ray handles GPU allocation automatically
✅ **Multi-node support** - Deploy across multiple machines
✅ **Auto-scaling** - Scale replicas based on load (future)
✅ **Better monitoring** - Ray dashboard shows detailed metrics
✅ **Isolation** - Deployment runs in managed Ray actors

### vs Container Mode

✅ **No container runtime needed** - Works without Docker/Podman
✅ **Native Python** - Direct vLLM integration, no containerization overhead
✅ **Better debugging** - Direct access to Python stack traces
✅ **Flexible deployment** - Easier to customize and extend

## Troubleshooting

### Issue: "Ray mode not available"

**Cause**: Ray not installed

**Solution**:
```bash
pip install vllm-playground[ray]
```

### Issue: "Failed to initialize Ray"

**Cause**: Ray cluster issues or network problems

**Solution**:
```bash
# Clean up Ray state
ray stop
rm -rf /tmp/ray

# Start fresh cluster
cd vllm_playground/ray_serve
python launch_cluster.py start
```

### Issue: "Cannot connect to Ray cluster"

**Cause**: Wrong address or cluster not running

**Solution**:
1. Check cluster is running: `ray status`
2. Verify address: `python launch_cluster.py get-address`
3. Update Ray Address in UI to match

### Issue: "Deployment timeout"

**Cause**: Model download or initialization taking longer than expected

**Solution**:
- Check Ray dashboard (http://localhost:8265) for progress
- Check vLLM Playground logs for download status
- Model may still be initializing - wait a bit longer
- For large models, this is normal (can take 5-10 minutes)

### Issue: "Out of memory"

**Cause**: Model too large for available GPU memory

**Solution**:
- Reduce `gpu_memory_utilization` (try 0.7 instead of 0.9)
- Use smaller model
- Increase tensor parallelism (use more GPUs)
- Check Ray dashboard for actual memory usage

## Backend Implementation

The Ray integration is implemented in:

- **Backend**: [vllm_playground/ray_serve/](vllm_playground/ray_serve/)
  - `ray_backend.py` - Main orchestrator
  - `engines/vllm_engine.py` - vLLM deployment
  - `engines/vllm_config.py` - Configuration builder

- **UI Integration**: [vllm_playground/app.py](vllm_playground/app.py)
  - Ray backend import and initialization
  - VLLMConfig model with Ray mode support
  - Ray mode validation in `/api/start` endpoint
  - Ray deployment handlers

## Next Steps

- **Documentation**: See [ray_serve/README.md](vllm_playground/ray_serve/README.md) for Ray backend details
- **Examples**: Check [examples/ray_deployment_example.py](examples/ray_deployment_example.py) for programmatic usage

## See Also

- [Ray Backend README](vllm_playground/ray_serve/README.md) - Ray backend architecture
- [Cluster Setup Guide](vllm_playground/ray_serve/CLUSTER_SETUP.md) - Ray cluster configuration
- [Quick Start](vllm_playground/ray_serve/QUICKSTART.md) - Getting started quickly
- [Ray Documentation](https://docs.ray.io/en/latest/serve/index.html) - Official Ray Serve docs
