# Accessing Gated Models in vLLM Playground

## What are Gated Models?

Gated models are models that require approval from the model authors before you can download and use them. They typically require:
1. Accepting terms of service on HuggingFace
2. Providing a HuggingFace access token

## CPU-Friendly Models in WebUI

The WebUI includes these excellent CPU-optimized models:

### 1. TinyLlama-1.1B-Chat-v1.0 (Ungated) ⭐ **Recommended for Testing**
- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: 1.1B parameters (~2.2GB)
- **Speed on CPU**: Fast (10-30 tokens/sec on modern CPUs)
- **No HF token required**
- **Best for**: Quick testing, development, limited hardware

### 2. Llama 3.2 1B (Gated) ⚠️ **Requires HF Token**
- **Model**: `meta-llama/Llama-3.2-1B`
- **Size**: 1B parameters (~2GB)
- **Speed on CPU**: Fast (10-30 tokens/sec)
- **Requires**: HuggingFace account + token
- **Best for**: Latest Meta technology, good quality responses

### 3. Gemma 2 2B (Gated) ⚠️ **Requires HF Token**
- **Model**: `google/gemma-2-2b`
- **Size**: 2B parameters (~4GB)
- **Speed on CPU**: Moderate (5-15 tokens/sec)
- **Requires**: HuggingFace account + token
- **Best for**: Google's latest efficient model, good quality

## How to Access Gated Models

### Step 1: Create HuggingFace Account

1. Go to https://huggingface.co/join
2. Create a free account
3. Verify your email

### Step 2: Request Access to Gated Models

#### For Llama 3.2:
1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B
2. Click "Agree and access repository"
3. Read and accept Meta's license agreement
4. Wait for approval (usually instant, but can take up to 24 hours)

#### For Gemma 2:
1. Visit https://huggingface.co/google/gemma-2-2b
2. Click "Agree and access repository"
3. Read and accept Google's terms
4. Wait for approval (usually instant)

### Step 3: Generate HuggingFace Access Token

1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "vLLM Playground")
4. Select token type:
   - **Read**: Sufficient for downloading models
   - **Write**: Not needed for vLLM
5. Click "Generate token"
6. **Copy the token immediately** (you won't be able to see it again!)

Example token format: `hf_ABcDEfGHiJKlMNoPQrSTuVwXyZ1234567890`

### Step 4: Configure Token in WebUI

You have three options:

#### Option A: Enter Token in WebUI (Recommended)
1. In the WebUI, go to the configuration panel
2. Find the "HuggingFace Token" field
3. Paste your token: `hf_xxxxxxxxxxxxx`
4. Select your gated model
5. Click "Start Server"

#### Option B: Set Environment Variable
```bash
export HF_TOKEN="hf_xxxxxxxxxxxxx"
python app.py
```

#### Option C: Use HuggingFace CLI
```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Login (saves token permanently)
huggingface-cli login

# Paste your token when prompted
```

Once logged in via CLI, you don't need to provide the token in WebUI.

## Testing Access

### Test if you have access:
```bash
# Check if you can access the model info
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.model_info('meta-llama/Llama-3.2-1B'))"
```

If this works, you have proper access!

## Example Configurations

### TinyLlama (No Token Required)
```json
{
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "max_model_len": 2048,
  "cpu_kvcache_space": 4,
  "dtype": "bfloat16"
}
```

### Llama 3.2 1B (Token Required)
```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "max_model_len": 2048,
  "cpu_kvcache_space": 4,
  "dtype": "bfloat16",
  "hf_token": "hf_xxxxxxxxxxxxx"
}
```

### Gemma 2 2B (Token Required)
```json
{
  "model": "google/gemma-2-2b",
  "max_model_len": 2048,
  "cpu_kvcache_space": 6,
  "dtype": "bfloat16",
  "hf_token": "hf_xxxxxxxxxxxxx"
}
```

## Troubleshooting

### Error: "Repository not found" or "Access denied"
**Cause**: You haven't requested access to the gated model yet.

**Solution**:
1. Visit the model page on HuggingFace
2. Click "Agree and access repository"
3. Wait for approval (check your email)
4. Try again

### Error: "Invalid token"
**Cause**: Token is incorrect, expired, or doesn't have read permissions.

**Solution**:
1. Generate a new token with "read" permissions
2. Make sure you copied the entire token (starts with `hf_`)
3. Update your configuration

### Error: "401 Unauthorized"
**Cause**: Token not provided or not recognized by vLLM.

**Solution**:
1. Make sure token is set in WebUI or as environment variable
2. Try logging in with `huggingface-cli login` first
3. Restart the WebUI

### Model downloads very slowly
**Cause**: Large model files + slow internet connection.

**Solution**:
1. Be patient - first download takes time
2. Models are cached locally after first download
3. Use `download_dir` to specify custom cache location
4. Consider downloading manually first:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-3.2-1B')"
   ```

## Security Best Practices

### ⚠️ Protect Your Token

1. **Never commit tokens to git**
   - Add `.env` to `.gitignore`
   - Don't paste tokens in public code

2. **Use environment variables for production**
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxxx"
   ```

3. **Rotate tokens regularly**
   - Generate new tokens every few months
   - Revoke old tokens you're not using

4. **Use separate tokens for different projects**
   - Easier to track and revoke if compromised

5. **Don't share tokens**
   - Each user should have their own token
   - Tokens are tied to your HuggingFace account

## Performance Comparison (CPU)

On a typical M2 MacBook Pro or modern Intel i7:

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| TinyLlama 1.1B | 2.2GB | 20-30 tok/s | Good | Testing, Development |
| Llama 3.2 1B | 2GB | 15-25 tok/s | Better | Quality + Speed balance |
| Gemma 2 2B | 4GB | 10-15 tok/s | Best | Best quality in 2B class |
| OPT 125M | 250MB | 50-100 tok/s | Basic | Quick testing only |

## Model Recommendations by Use Case

### For Learning/Testing
- **TinyLlama 1.1B** - No token required, fast, good enough

### For Development
- **Llama 3.2 1B** - Good quality, fast, requires token

### For Production (CPU)
- **Gemma 2 2B** - Best quality in small size, requires token

### For GPU
- **Llama 2 7B** or **Mistral 7B** - Much better quality, requires GPU

## Additional Resources

- HuggingFace Models: https://huggingface.co/models
- HuggingFace Tokens: https://huggingface.co/settings/tokens
- Llama 3.2 Model Card: https://huggingface.co/meta-llama/Llama-3.2-1B
- Gemma 2 Model Card: https://huggingface.co/google/gemma-2-2b
- TinyLlama Model Card: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

## Quick Start Command

```bash
# Set your token
export HF_TOKEN="hf_xxxxxxxxxxxxx"

# Start WebUI
python app.py

# In WebUI, select:
# - Model: meta-llama/Llama-3.2-1B
# - Max Model Length: 2048
# - CPU KV Cache: 4
# - Click "Start Server"
```

## FAQ

**Q: Do I need to pay for HuggingFace?**
A: No, free account is sufficient for these models.

**Q: How long does approval take?**
A: Usually instant for Llama 3.2 and Gemma 2.

**Q: Can I use the same token for multiple models?**
A: Yes, one token works for all models you have access to.

**Q: What if I don't want to use gated models?**
A: Use TinyLlama or OPT models - they're ungated and work great!

**Q: Will my token expire?**
A: Tokens don't expire automatically, but you should rotate them periodically for security.

**Q: Can I use these models offline?**
A: Yes, after the first download, models are cached locally.
