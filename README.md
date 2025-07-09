# Live Transcription for Meetings

A real-time audio transcription system using FastAPI, Socket.IO, and Whisper for live meeting transcription. This application captures audio from your microphone and provides instant transcription using OpenAI's Whisper model.

## Features

- 🎤 **Real-time Audio Streaming**: Captures audio from microphone with Web Audio API
- 🗣️ **Live Transcription**: Uses FasterWhisper for efficient real-time speech-to-text
- 🔄 **Correction Buffer**: Implements chunked processing for improved accuracy
- 🌐 **WebSocket Communication**: Uses Socket.IO for low-latency real-time communication
- 📝 **Auto-correction**: Processes audio in chunks to provide corrected transcriptions
- 🎯 **Voice Activity Detection (VAD)**: Filters out non-speech audio segments

## Architecture

### Backend (app.py)
- **FastAPI**: Web framework with ASGI support
- **Socket.IO**: Real-time bidirectional communication
- **FasterWhisper**: Efficient Whisper implementation for transcription
- **CorrectionBuffer**: Handles audio chunking and processing

### Frontend (index.html)
- **Web Audio API**: Captures microphone audio
- **AudioWorklet**: Processes and resamples audio to 16kHz
- **Socket.IO Client**: Communicates with backend
- **Real-time Display**: Shows transcription results instantly

### Audio Processing (audio-processor.js)
- **PCM Resampler**: Converts audio to 16kHz sample rate
- **Buffering**: Manages audio chunks for streaming
- **Worklet Processor**: Handles audio processing in separate thread

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd live-transcription-meetings
```

2. **Install Python dependencies**:
```bash
pip install fastapi uvicorn python-socketio numpy faster-whisper
```

3. **Install PyTorch with CUDA support** (for GPU acceleration):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

1. **Start the server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:1090
```

3. **Grant microphone permissions** when prompted

4. **Click "Start Streaming"** to begin transcription

## Configuration

### Model Settings
- **Model Size**: Default is "medium" (can be changed to "tiny", "base", "small", "large")
- **Language**: Auto-detection (can be set to specific language)
- **Device**: CUDA for GPU acceleration (falls back to CPU)
- **Compute Type**: float16 for better performance

### Audio Settings
- **Sample Rate**: 16kHz (required by Whisper)
- **Chunk Size**: 8 seconds for correction buffer
- **Buffer Size**: 4096 samples for audio processing

## Technical Details

### Audio Processing Pipeline
1. **Capture**: Web Audio API captures microphone input
2. **Resample**: AudioWorklet resamples to 16kHz
3. **Stream**: Socket.IO sends audio chunks to backend
4. **Buffer**: CorrectionBuffer accumulates audio data
5. **Transcribe**: FasterWhisper processes audio chunks
6. **Emit**: Transcription results sent back to frontend

### Correction System
- Processes audio in overlapping chunks for better accuracy
- Uses async locks to prevent race conditions
- Handles remaining audio on stream finish
- Provides both interim and final transcriptions

## API Endpoints

### Socket.IO Events

#### Client to Server
- `audio_data`: Send audio chunk for transcription
- `finish_streaming`: Signal end of audio stream

#### Server to Client
- `transcription`: Receive transcription results
  - `type`: "correction" or "final"
  - `start`: Start time in seconds
  - `end`: End time in seconds
  - `text`: Transcribed text

## Performance Optimization

- **GPU Acceleration**: Uses CUDA when available
- **Async Processing**: Non-blocking audio processing
- **Thread Pool**: Parallel processing for multiple clients
- **VAD Filter**: Reduces processing of non-speech audio
- **Chunked Processing**: Optimized for real-time performance

## Browser Compatibility

- **Chrome**: Full support
- **Firefox**: Full support
- **Safari**: Full support (requires HTTPS for microphone access)
- **Edge**: Full support

## Security Notes

- Requires HTTPS for microphone access in production
- CORS enabled for all origins (configure for production)
- No audio data is stored permanently
- Real-time processing only

## Troubleshooting

### Common Issues

1. **Microphone not working**: Check browser permissions
2. **Poor transcription quality**: Ensure good audio quality and minimize background noise
3. **High CPU usage**: Consider using GPU acceleration or smaller model
4. **Connection issues**: Check firewall settings and network connectivity

### Performance Tips

- Use headphones to reduce echo
- Speak clearly and at moderate pace
- Ensure stable internet connection
- Close unnecessary applications to free up resources

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 1090
```

### Testing
- Test with different microphones
- Verify transcription accuracy
- Check performance under load
- Test connection stability

## License

This project is open source and available under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please open an issue in the GitHub repository.
