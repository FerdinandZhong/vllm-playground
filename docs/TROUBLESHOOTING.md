# Troubleshooting Guide for vLLM Playground

## Common Issues and Solutions

### 1. Engine Core Initialization Failed with "Torch not compiled with CUDA enabled"

**Error:**
```
AssertionError: Torch not compiled with CUDA enabled
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_DP0': 1}
```

**Root Cause:**
vLLM is trying to use CUDA/GPU mode on macOS where CUDA is not available. This happens when the `--device cpu` flag is not explicitly set.

**Solution:**
This is now fixed automatically in the WebUI - it will detect macOS and add the `--device cpu` flag. However, if you're running vLLM manually or seeing this error:

1. **Ensure you're using CPU mode** - The WebUI auto-detects macOS
2. **Verify the command includes** `--device cpu`
3. **Check your vLLM version** - Make sure you have vLLM with CPU backend support
4. **Environment variables are set:**
   ```bash
   export VLLM_CPU_KVCACHE_SPACE=4
   export VLLM_CPU_OMP_THREADS_BIND=auto
   export VLLM_CPU_MOE_PREPACK=0
   export VLLM_CPU_SGL_KERNEL=0
   ```

### 2. Engine Core Initialization Failed (Memory Issues)

**Error:**
```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore_DP0': 1}
```
(Without the CUDA error)

**Solution:**
- Use a smaller model (e.g., `facebook/opt-125m` or `facebook/opt-350m` for testing)
- Reduce `max_model_len` to 512, 1024, or 2048
- Reduce KV cache space in settings (try 2-4 GB instead of 40 GB)

#### B. Model Not Compatible with CPU Backend
Some models may not work well with vLLM's CPU backend.

**Solution:**
- Try a different model from the OPT family first: `facebook/opt-125m`, `facebook/opt-350m`
- Check if the model supports CPU inference

#### C. CPU Optimization Issues on Apple Silicon
Some CPU optimizations may cause issues on M1/M2/M3 Macs (now automatically disabled).

**Solution:**
The WebUI now automatically disables problematic optimizations. If running manually, add these to your environment or `config/vllm_cpu.env`:
```bash
export VLLM_CPU_MOE_PREPACK=0
export VLLM_CPU_SGL_KERNEL=0
```

### 3. max_num_batched_tokens Error

**Error:**
```
Value error, max_num_batched_tokens (2048) is smaller than max_model_len (131072)
```

**Solution:**
This is now fixed automatically, but if you still see it:
- Explicitly set `max_model_len` to a reasonable value (2048, 4096, or 8192)
- The WebUI will automatically set `max_num_batched_tokens` to match

### 3. Memory Issues - OOM (Out of Memory)

**Symptoms:**
- Process crashes
- System becomes unresponsive
- "Cannot allocate memory" errors

**Solution:**
1. **Reduce model size:** Use smaller models
2. **Reduce max_model_len:** Try 512, 1024, or 2048
3. **Reduce KV cache:** Set to 2-4 GB
4. **Close other applications:** Free up RAM

**Conservative Settings for CPU:**
```json
{
  "model": "facebook/opt-125m",
  "max_model_len": 1024,
  "cpu_kvcache_space": 2,
  "dtype": "bfloat16"
}
```

### 4. Model Download Issues

**Symptoms:**
- Timeout errors
- Connection refused
- Model not found

**Solution:**
1. Check your internet connection
2. For gated models (Llama 2, etc.), ensure you have:
   - Hugging Face account
   - Accepted model terms
   - Set HF_TOKEN environment variable
3. Pre-download models:
   ```bash
   python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('facebook/opt-125m')"
   ```

### 5. Server Won't Start

**Solution:**
1. Check if port is already in use:
   ```bash
   lsof -i :8000
   ```
2. Try a different port in settings
3. Check logs in WebUI for specific errors

### 6. Slow Performance on CPU

**Expected Behavior:**
CPU inference is inherently slower than GPU. Typical speeds:
- Small models (125M-350M): 10-50 tokens/second
- Medium models (1B-3B): 1-10 tokens/second
- Large models (7B+): 0.1-2 tokens/second

**Optimization:**
1. Increase KV cache (if you have RAM): 10-40 GB
2. Reduce max_tokens in generation
3. Use smaller models
4. Ensure no other heavy processes are running

## Recommended Starting Configuration

### For Testing (Minimal Resources)
```json
{
  "model": "facebook/opt-125m",
  "max_model_len": 1024,
  "cpu_kvcache_space": 2,
  "dtype": "bfloat16"
}
```

### For Development (Moderate Resources)
```json
{
  "model": "facebook/opt-350m",
  "max_model_len": 2048,
  "cpu_kvcache_space": 4,
  "dtype": "bfloat16"
}
```

### For Production (High Resources)
```json
{
  "model": "facebook/opt-1.3b",
  "max_model_len": 4096,
  "cpu_kvcache_space": 10,
  "dtype": "bfloat16"
}
```

## Getting More Debug Information

To see detailed error messages:

1. **In WebUI:** Check the Server Logs panel
2. **From command line:**
   ```bash
   python app.py
   ```
   Then check terminal output

3. **Enable verbose logging:**
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   python app.py
   ```

## macOS-Specific Issues

### Apple Silicon (M1/M2/M3) Compatibility
- vLLM CPU backend works but may be slower
- Some optimizations may need to be disabled
- Intel-based Macs may have different behavior

### Environment Setup
Make sure you have:
```bash
# Check Python version (3.8+)
python --version

# Check if vLLM is installed
python -c "import vllm; print(vllm.__version__)"

# Check available memory
sysctl hw.memsize
```

## Still Having Issues?

1. **Check the full error log** - The root cause is usually shown above the final error
2. **Try the smallest model first** - `facebook/opt-125m` with minimal settings
3. **Monitor system resources** - Use Activity Monitor to check RAM usage
4. **Check vLLM compatibility** - Some features may not work on CPU backend

## Reporting Issues

When reporting issues, please include:
1. Full error log from WebUI
2. Your system specs (macOS version, RAM, CPU)
3. Model and configuration you're trying to use
4. Steps to reproduce the error

