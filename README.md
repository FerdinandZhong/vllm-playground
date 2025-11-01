# 🚀 vLLM WebUI

A modern, feature-rich web interface for managing and interacting with vLLM (Virtual Large Language Model) servers.

![vLLM WebUI](https://img.shields.io/badge/vLLM-WebUI-blue)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)

## ✨ Features

### 🎛️ Server Configuration
- **Easy Model Selection**: Choose from popular models or specify custom model names
- **Advanced Settings**: Configure tensor parallelism, GPU memory utilization, dtype, and more
- **One-Click Control**: Start and stop vLLM servers with a single click
- **Real-time Status**: Monitor server status and uptime

### 💬 Chat Interface
- **Interactive Chat**: Test your models with a beautiful chat interface
- **Configurable Parameters**: Adjust temperature, max tokens, and other generation parameters on the fly
- **Chat History**: Maintains conversation context for multi-turn interactions
- **Streaming Support**: Real-time response streaming (coming soon)

### 📋 Log Viewer
- **Real-time Logs**: Watch server logs stream in real-time via WebSocket
- **Log Filtering**: Categorized logs (info, warning, error)
- **Auto-scroll**: Option to automatically follow new log entries
- **Log Management**: Clear logs when needed

### 🎨 Modern UI
- **Dark Theme**: Easy on the eyes with a beautiful dark interface
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Animations**: Polished interactions and transitions
- **Status Indicators**: Visual feedback for all operations

## 📋 Prerequisites

- Python 3.8 or higher
- vLLM installed (`pip install vllm`)
- CUDA-compatible GPU (for running vLLM)
- Modern web browser

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/vllm-webui.git
cd vllm-webui
```

2. Run the automated installation script:
```bash
./install.sh
```

Or install manually:

```bash
pip install -r requirements.txt
```

### Running the WebUI

**Option 1: Quick Start Script (Recommended)**

```bash
./start.sh
```

**Option 2: Manual Start**

```bash
python app.py
```

**Next Steps:**

1. Open your browser and navigate to:
```
http://localhost:7860
```

2. Configure your vLLM server settings in the left panel

3. Click "Start Server" to launch vLLM

4. Once the server is running, use the chat interface to interact with your model

### Custom Port

You can specify a custom port for the WebUI:
```bash
WEBUI_PORT=8080 python app.py
```

## 🎯 Usage Guide

### Starting a vLLM Server

1. **Select a Model**:
   - Choose from the dropdown list of popular models
   - Or enter a custom model name in the text field below

2. **Configure Settings**:
   - **Host**: Server host address (default: 0.0.0.0)
   - **Port**: Server port (default: 8000)
   - **Tensor Parallel Size**: Number of GPUs to use for tensor parallelism
   - **GPU Memory Utilization**: Percentage of GPU memory to use (default: 90%)
   - **Data Type**: Precision for model weights (auto, float16, bfloat16, float32)
   - **Max Model Length**: Maximum sequence length (optional)

3. **Advanced Options**:
   - ☑️ Trust Remote Code: Allow execution of code from model repository
   - ☑️ Enable Prefix Caching: Enable KV cache reuse
   - ☑️ Disable Log Stats: Disable periodic stats logging

4. **Start**: Click "▶️ Start Server" button

### Using the Chat Interface

1. Wait for the server to start (watch the logs panel)

2. Once running, type your message in the chat input area

3. Adjust generation parameters:
   - **Temperature**: Controls randomness (0.0 = deterministic, 2.0 = very random)
   - **Max Tokens**: Maximum length of the response

4. Press "Send" or `Ctrl+Enter` to send your message

5. The model's response will appear in the chat area

6. Continue the conversation with context maintained

### Monitoring Logs

- **Real-time Updates**: Logs stream automatically via WebSocket
- **Auto-scroll**: Toggle auto-scroll to follow new entries
- **Clear Logs**: Use the "Clear" button to reset the log view
- **Log Types**:
  - 🔵 Info: General information
  - 🟡 Warning: Non-critical issues
  - 🔴 Error: Critical errors
  - 🟢 Success: Successful operations

## 🏗️ Architecture

### Backend (FastAPI)
- **API Server**: RESTful endpoints for server control
- **WebSocket**: Real-time log streaming
- **Process Management**: Manages vLLM server subprocess
- **Proxy**: Routes chat requests to vLLM server

### Frontend (HTML/CSS/JavaScript)
- **Responsive UI**: Modern, mobile-friendly interface
- **Real-time Updates**: WebSocket client for live logs
- **AJAX Requests**: Async communication with backend
- **State Management**: Client-side state tracking

## 📁 Project Structure

```
webui/
├── app.py                  # FastAPI backend server
├── index.html              # Main HTML interface
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── static/
    ├── css/
    │   └── style.css      # Styles and themes
    └── js/
        └── app.js         # Frontend JavaScript
```

## 🔧 Configuration

### Environment Variables

- `WEBUI_PORT`: Port for WebUI server (default: 7860)

### vLLM Configuration

All vLLM server configurations are done through the web interface. The following parameters are supported:

- Model name
- Host and port
- Tensor parallel size
- GPU memory utilization
- Data type
- Max model length
- Trust remote code
- Prefix caching
- Log stats

## 🐛 Troubleshooting

### Server Won't Start

1. Check the logs panel for error messages
2. Verify the model name is correct
3. Ensure you have enough GPU memory
4. Check that the port is not already in use

### WebSocket Connection Failed

1. Check if the WebUI server is running
2. Verify firewall settings
3. Try refreshing the page

### Chat Not Working

1. Ensure the vLLM server is running (check status indicator)
2. Wait for the model to fully load (watch logs)
3. Check that the server port matches configuration

### Out of Memory

1. Reduce GPU memory utilization percentage
2. Use a smaller model
3. Increase tensor parallel size to distribute across more GPUs
4. Reduce max model length

## 🎨 Customization

### Changing Theme Colors

Edit `static/css/style.css` and modify the CSS variables in `:root`:

```css
:root {
    --primary-color: #4f46e5;
    --bg-primary: #0f172a;
    /* ... more variables */
}
```

### Adding New Models

Edit `app.py` and add to the `list_models()` function:

```python
{
    "name": "your-org/your-model",
    "size": "7B",
    "description": "Your model description"
}
```

## 🚧 Roadmap

- [ ] Streaming chat responses
- [ ] Multiple chat sessions
- [ ] Save/load chat history
- [ ] Model comparison mode
- [ ] API endpoint testing
- [ ] Performance metrics dashboard
- [ ] User authentication
- [ ] Multi-user support
- [ ] Configuration presets
- [ ] Export logs to file

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

This project is built to work with [vLLM](https://github.com/vllm-project/vllm).

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [vLLM](https://github.com/vllm-project/vllm)
- Icons from emoji

## 📧 Support

For issues and questions:
- Open an issue on this repository
- Check the [vLLM documentation](https://docs.vllm.ai/)
- Visit the [vLLM GitHub repository](https://github.com/vllm-project/vllm)

---

Made with ❤️ for the vLLM community

