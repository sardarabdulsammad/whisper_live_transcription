# Simple Live Whisper Transcription with Translation

A **simple** real-time audio transcription and translation system using FastAPI, Socket.IO, and Whisper. This straightforward implementation provides live meeting transcription with instant translation support.

## Key Features

- 🎤 **Simple Audio Streaming**: Basic microphone capture with Web Audio API
- 🗣️ **Streaming Whisper**: Real-time speech-to-text using FasterWhisper
- 🌍 **Live Translation**: Instant translation to any supported language
- ⚙️ **Easy Configuration**: Simple chunk size and model adjustments
- 🔧 **Minimal Setup**: Just a few files, easy to understand and modify

## What Makes This Simple

This implementation focuses on **simplicity over complexity**:
- **Single main file** (`app.py`) handles everything
- **Configurable chunk size** (default: 8 seconds)
- **Easy translation model switching** for different languages
- **No complex buffering** - straightforward audio processing
- **Minimal dependencies** - just the essentials

## Quick Configuration

### Adjust Chunk Size
Change processing chunk size in `app.py`:
```python
# Default is 8 seconds - you can change this
sio.correction_buffers[sid] = CorrectionBuffer(audio_model, chunk_size=8)
```
- **Smaller chunks** (2-4s): Faster response, less accuracy
- **Larger chunks** (10-15s): Better accuracy, slower response

### Change Translation Language
Modify the translation model in `translator.py` for your target language:
```python
# Current: English to Chinese
self.trg = "zh"  # Change to your target language
self.model_name = f"Helsinki-NLP/opus-mt-{self.src}-{self.trg}"
```

**Supported language codes:**
- `zh` - Chinese
- `es` - Spanish  
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `ru` - Russian
- `ar` - Arabic
- And many more...

## Installation & Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the application**:
```bash
python app.py
```

3. **Open browser**: Go to `http://localhost:1090`

4. **Start transcribing**: Click "Start Streaming" and speak!

## How It Works (Simple Overview)

1. **Audio Capture**: Browser captures microphone audio
2. **Audio Streaming**: Audio chunks sent via WebSocket to server
3. **Whisper Processing**: FasterWhisper transcribes audio chunks
4. **Translation**: Accumulated text gets translated in real-time
5. **Display**: Results shown instantly in browser

## File Structure

```
whisper_live_transcription-translation/
├── app.py                 # Main server (FastAPI + Socket.IO)
├── translator.py          # Simple translation module
├── audio-processor.js     # Audio processing worklet
├── index.html            # Simple web interface
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Simple Customization Examples

### Change Whisper Model Size
In `app.py`:
```python
# Faster but less accurate
audio_model = FasterWhisperASR(model_size="small")

# More accurate but slower
audio_model = FasterWhisperASR(model_size="large")
```

### Add New Translation Language
In `translator.py`, create a new translator class:
```python
class EnglishToSpanishTranslator:
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-es"
        # ...rest is the same
```

### Modify Chunk Processing
In `app.py`, adjust the `CorrectionBuffer`:
```python
# Process every 5 seconds instead of 8
sio.correction_buffers[sid] = CorrectionBuffer(audio_model, chunk_size=5)
```

## Configuration Options

### Audio Settings
- **Chunk Size**: 2-15 seconds (default: 8)
- **Sample Rate**: 16kHz (fixed for Whisper)
- **Model Size**: tiny, base, small, medium, large

### Translation Settings
- **Source Language**: Auto-detect or specify
- **Target Language**: Any Helsinki-NLP supported language
- **Translation Model**: Marian MT models

## Requirements

- Python 3.8+
- Modern web browser
- Microphone access
- Optional: CUDA for GPU acceleration

## Performance Notes

- **GPU recommended** for larger Whisper models
- **Smaller chunks** = faster response but less context
- **Larger chunks** = better accuracy but more delay
- **Translation** adds ~100-500ms processing time

## Troubleshooting

1. **No transcription**: Check microphone permissions
2. **Slow processing**: Try smaller Whisper model or reduce chunk size
3. **Translation errors**: Verify Helsinki-NLP model for your language pair
4. **Connection issues**: Check WebSocket connection and firewall

## Simple Modifications

This codebase is designed to be **easily modifiable**:

- Want different language? Change model in `translator.py`
- Need faster response? Reduce chunk size in `app.py`
- Want better accuracy? Use larger Whisper model
- Need custom UI? Modify `index.html`

## License

MIT License - Simple and permissive for easy use and modification.

---

*This is intentionally a simple implementation. For production use, consider adding error handling, authentication, and performance optimizations.*
