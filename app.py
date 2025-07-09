from fastapi import FastAPI
import socketio
import numpy as np
import logging
import time
import sys
import uvicorn
from functools import lru_cache
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
from translator import EnglishToChineseTranslator, TranslationBuffer

# Suppress verbose INFO logs
logging.basicConfig(level=logging.ERROR)

# Setup Socket.IO with ASGI support
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)

# Global thread pool for ASR processing
executor = ThreadPoolExecutor(max_workers=4)

@lru_cache(10**6)
class FasterWhisperASR:
    """FasterWhisper ASR implementation"""
    
    sep = ""
    
    def __init__(self, model_size="medium", language="auto", device="cuda", compute_type="float16"):
        self.model_size = model_size
        self.original_language = language if language != "auto" else None
        self.device = device
        self.compute_type = compute_type
        self.transcribe_kargs = {}
        self.model = self._load_model()
        
    def _load_model(self):
        from faster_whisper import WhisperModel
        
        print(f"Loading FasterWhisper {self.model_size} model...")
        start_time = time.time()
        model = WhisperModel(
            self.model_size, 
            device=self.device, 
            compute_type=self.compute_type
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        return model
    
    def transcribe(self, audio):
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            beam_size=3,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs
        )
        return list(segments)
    
    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o
    
    def segments_end_ts(self, res):
        return [s.end for s in res]
    
    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

class CorrectionBuffer:
    """Handles larger chunks for correction"""
    
    SAMPLING_RATE = 16000

    def __init__(self, asr_model, chunk_size=8, logfile=sys.stderr):
        self.asr = asr_model
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_samples = 0  # Track how much audio we've already processed
        self.logfile = logfile
        self.CHUNK_SIZE = chunk_size
        self.processing_lock = asyncio.Lock()  # Add this line
        
    def add_audio(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)
        
    def should_process(self):
        unprocessed_samples = len(self.audio_buffer) - self.processed_samples
        buffer_seconds = unprocessed_samples / self.SAMPLING_RATE
        return buffer_seconds >= self.CHUNK_SIZE
        
       
    async def process_chunk_async(self):
        """Async wrapper for process_chunk with lock"""
        async with self.processing_lock:
            if not self.should_process():
                return None
                
            chunk_samples = int(self.CHUNK_SIZE * self.SAMPLING_RATE)
            chunk_start = self.processed_samples
            chunk_end = chunk_start + chunk_samples
            
            # Mark as processing immediately
            self.processed_samples = chunk_end
            
            # Only process the new chunk, not overlapping with previous
            chunk_to_process = self.audio_buffer[chunk_start:chunk_end]
            
            # Process directly without context for speed
            segments = self.asr.transcribe(chunk_to_process)
            full_text = " ".join([s.text for s in segments])
            
            return {
                "start_time": chunk_start / self.SAMPLING_RATE,
                "end_time": chunk_end / self.SAMPLING_RATE,
                "text": full_text
            }
        
    def get_remaining_audio(self):
        """Get only the audio that hasn't been processed yet"""
        if self.processed_samples < len(self.audio_buffer):
            return self.audio_buffer[self.processed_samples:]
        return np.array([], dtype=np.float32)

# Initialize ASR pipeline
audio_model = FasterWhisperASR(model_size="medium")
audio_model.use_vad()

# Initialize translation
translator = EnglishToChineseTranslator()

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    print(f'Client connected: {sid}')

@sio.event
async def disconnect(sid):
    print(f'Client disconnected: {sid}')
    # Clean up translation buffers
    if hasattr(sio, 'translation_buffers') and sid in sio.translation_buffers:
        del sio.translation_buffers[sid]
    
@sio.event
async def audio_data(sid, data):
    # Decode audio data
    if isinstance(data, str):
        binary_data = base64.b64decode(data)
        audio = np.frombuffer(binary_data, dtype=np.float32)
    else:
        audio = np.frombuffer(data, dtype=np.float32)
        
    # Initialize correction buffer
    if not hasattr(sio, 'correction_buffers'):
        sio.correction_buffers = {}
        
    if sid not in sio.correction_buffers:
        sio.correction_buffers[sid] = CorrectionBuffer(audio_model, chunk_size=8)
        
    # Initialize translation buffer
    if not hasattr(sio, 'translation_buffers'):
        sio.translation_buffers = {}
        
    if sid not in sio.translation_buffers:
        sio.translation_buffers[sid] = TranslationBuffer(translator)
        
    # Add audio to buffer
    sio.correction_buffers[sid].add_audio(audio)
    
    # Process in background
    asyncio.create_task(process_audio(sid, sio.correction_buffers[sid]))

async def process_audio(sid, correction_buffer):
    """Process audio in background threads"""
    
    # Process correction buffer with lock
    correction = await correction_buffer.process_chunk_async()
    if correction and correction["text"].strip():
        # Emit transcription
        await sio.emit('transcription', {
            "type": "correction",
            "start": float(correction["start_time"]),
            "end": float(correction["end_time"]),
            "text": correction["text"]
        }, to=sid)
        
        # Process translation
        if hasattr(sio, 'translation_buffers') and sid in sio.translation_buffers:
            loop = asyncio.get_event_loop()
            translated_text = await loop.run_in_executor(
                executor,
                sio.translation_buffers[sid].add_text,
                correction["text"]
            )
            
            if translated_text.strip():
                await sio.emit('translation', {
                    "type": "correction",
                    "start": float(correction["start_time"]),
                    "end": float(correction["end_time"]),
                    "text": translated_text,
                    "accumulated_original": sio.translation_buffers[sid].get_accumulated_text()
                }, to=sid)

@sio.event
async def finish_streaming(sid):
    if hasattr(sio, 'correction_buffers') and sid in sio.correction_buffers:
        # Process any remaining audio that hasn't been processed yet
        remaining_audio = sio.correction_buffers[sid].get_remaining_audio()
        if len(remaining_audio) > 0:
            loop = asyncio.get_event_loop()
            segments = await loop.run_in_executor(
                executor,
                audio_model.transcribe,
                remaining_audio
            )
            final_text = " ".join([s.text for s in segments])
            if final_text.strip():
                start_time = sio.correction_buffers[sid].processed_samples / 16000
                end_time = start_time + len(remaining_audio) / 16000
                
                # Emit final transcription
                await sio.emit('transcription', {
                    "type": "final",
                    "start": start_time,
                    "end": end_time,
                    "text": final_text
                }, to=sid)
                
                # Emit final translation
                if hasattr(sio, 'translation_buffers') and sid in sio.translation_buffers:
                    final_translation = await loop.run_in_executor(
                        executor,
                        sio.translation_buffers[sid].add_text,
                        final_text
                    )
                    
                    if final_translation.strip():
                        await sio.emit('translation', {
                            "type": "final",
                            "start": start_time,
                            "end": end_time,
                            "text": final_translation,
                            "accumulated_original": sio.translation_buffers[sid].get_accumulated_text()
                        }, to=sid)
        
        # Cleanup
        del sio.correction_buffers[sid]
        if hasattr(sio, 'translation_buffers') and sid in sio.translation_buffers:
            del sio.translation_buffers[sid]

# Mount the Socket.IO app
app.mount("/", socket_app)

if __name__ == '__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=1090)