# Chat Templates: Model-Specific vs Generic

## Overview

Different language models are trained with different chat formats (called "chat templates"). Using the correct chat template for your model can significantly improve response quality.

As of **November 4, 2025**, the vLLM Playground now **automatically detects your model** and applies the appropriate chat template!

## Why Chat Templates Matter

Chat templates tell the model how to interpret the conversation structure. Using the wrong template can cause:
- ❌ Nonsensical or hallucinated responses
- ❌ The model generating fake conversations
- ❌ Poor instruction following
- ❌ Responses that don't stop properly

Using the **correct template** results in:
- ✅ Coherent, relevant responses
- ✅ Better instruction following
- ✅ Proper conversation flow
- ✅ Clean response boundaries

## Supported Models

The WebUI automatically detects and applies templates for these model families:

### 1. **Llama 2/3 Models** (Meta)
- **Models**: `meta-llama/Llama-2-*`, `meta-llama/Llama-3-*`
- **Format**: `[INST] user message [/INST] assistant response </s>`
- **Stop Tokens**: `[INST]`, `</s>`, `<s>`
- **Example Models**:
  - `meta-llama/Llama-2-7b-chat-hf`
  - `meta-llama/Llama-3.2-1B`

### 2. **Mistral/Mixtral Models**
- **Models**: `mistralai/Mistral-*`, `mistralai/Mixtral-*`
- **Format**: `[INST] user message [/INST] assistant response </s>`
- **Stop Tokens**: `[INST]`, `</s>`
- **Example Models**:
  - `mistralai/Mistral-7B-Instruct-v0.2`
  - `mistralai/Mixtral-8x7B-Instruct`

### 3. **Gemma Models** (Google)
- **Models**: `google/gemma-*`
- **Format**: `<start_of_turn>user\nmessage<end_of_turn>\n<start_of_turn>model\nresponse`
- **Stop Tokens**: `<start_of_turn>`, `<end_of_turn>`
- **Example Models**:
  - `google/gemma-2-2b`
  - `google/gemma-7b`

### 4. **TinyLlama Models**
- **Models**: `TinyLlama/*`
- **Format**: ChatML-style with `<|user|>`, `<|assistant|>`, `<|system|>`
- **Stop Tokens**: `<|user|>`, `<|system|>`, `</s>`
- **Example Models**:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

### 5. **Vicuna Models**
- **Models**: `*vicuna*`
- **Format**: `USER: message\nASSISTANT: response`
- **Stop Tokens**: `USER:`, `</s>`

### 6. **Alpaca Models**
- **Models**: `*alpaca*`
- **Format**: `### Instruction:\nmessage\n\n### Response:\nresponse`
- **Stop Tokens**: `### Instruction:`, `### Response:`

### 7. **CodeLlama Models**
- **Models**: `codellama/*`, `*code-llama*`
- **Format**: Same as Llama 2 with `[INST]` tags
- **Stop Tokens**: `[INST]`, `</s>`, `<s>`
- **Example Models**:
  - `codellama/CodeLlama-7b-Instruct-hf`

### 8. **OPT Models** (Facebook)
- **Models**: `facebook/opt-*`
- **Format**: Simple `User: / Assistant:` format
- **Stop Tokens**: `User:`, `Assistant:`
- **Example Models**:
  - `facebook/opt-125m`
  - `facebook/opt-1.3b`

### 9. **Generic/Unknown Models**
- For models not in the above list, a generic template is used
- **Format**: `User: message\nAssistant: response`
- **Stop Tokens**: `User:`, `\nUser:`, `Assistant:`

## How It Works

### Automatic Detection

When you start the vLLM server, the WebUI:

1. **Analyzes the model name** you selected
2. **Matches it** to a known model family
3. **Applies the correct chat template** automatically
4. **Sets appropriate stop tokens** for that model

You'll see this in the logs:
```
[WEBUI] Using chat template for model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Code Location

The chat template detection is implemented in `app.py`:
- `get_chat_template_for_model(model_name)` - Returns the template
- `get_stop_tokens_for_model(model_name)` - Returns stop tokens

## Examples

### Example 1: TinyLlama
**Input**: "What is Python?"

**Without proper template** (generic):
```
User: What is Python?
Assistant: Python is a programming language Human: tell me more Assistant: ...
```
❌ Model starts generating fake conversation

**With TinyLlama template**:
```
<|user|>
What is Python?</s>
<|assistant|>
Python is a high-level, interpreted programming language known for...
```
✅ Clean, focused response

### Example 2: Llama 3
**Input**: "Write a haiku about coding"

**With Llama 3 template**:
```
[INST] Write a haiku about coding [/INST] Code flows like water,
Logic builds bridges of thought,
Bugs teach us patience.
```
✅ Proper response that stops cleanly

## Customizing Templates

### For Advanced Users

If you need to use a custom chat template, you can:

1. **Edit `app.py`** and modify the `get_chat_template_for_model()` function
2. **Add your model** to the detection logic
3. **Restart the WebUI**

Example:
```python
def get_chat_template_for_model(model_name: str) -> str:
    model_lower = model_name.lower()
    
    # Add your custom model
    if 'my-custom-model' in model_lower:
        return "{% for message in messages %}...your template...{% endfor %}"
    
    # ... rest of the function
```

### Testing a Template

You can test if a template works well:
1. Start the server with your model
2. Send a simple message: "Hello, how are you?"
3. Check if the response:
   - Is relevant (not hallucinated)
   - Stops cleanly (doesn't continue forever)
   - Doesn't generate fake conversation

## Troubleshooting

### Issue: Model still generates nonsense

**Solution**:
1. Check the server logs to see which template was applied
2. The model might need a different template
3. Try manually adding a template for your specific model in `app.py`

### Issue: Responses cut off too early

**Solution**:
- The stop tokens might be too aggressive
- Increase `max_tokens` in the chat interface
- Modify stop tokens in `get_stop_tokens_for_model()`

### Issue: Responses don't stop

**Solution**:
- Add more stop tokens specific to your model
- Check if the model was trained with special tokens
- Look at the model's HuggingFace card for the correct format

## References

- **Transformers Documentation**: https://huggingface.co/docs/transformers/chat_templating
- **vLLM Chat Templates**: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-template
- **Model Cards**: Check each model's HuggingFace page for its specific chat format

## Contributing

If you find a model that doesn't work well with the auto-detected template:

1. Check the model's HuggingFace card for its chat format
2. Add it to `get_chat_template_for_model()` in `app.py`
3. Test it thoroughly
4. Consider contributing back to the project!

---

**Last Updated**: November 4, 2025
**Status**: ✅ Automatic model-specific templates active

