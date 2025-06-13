import sounddevice as sd
import threading
import queue
import time
import gc
from collections import deque
from kokoro import KPipeline
import soundfile as sf
import numpy as np

class EfficientTTSStreamer:
    def __init__(self, lang_code='a', voice='af_bella', buffer_size=3, samplerate=24000):
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.samplerate = samplerate
        self.buffer_size = buffer_size
        
        # Audio playback queue and control
        self.audio_queue = queue.Queue(maxsize=buffer_size)
        self.playback_thread = None
        self.stop_playback = threading.Event()
        self.is_playing = False
        
        # Streaming state
        self.current_stream = None
        
    def _playback_worker(self):
        """Background thread for continuous audio playback"""
        while not self.stop_playback.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:  # Poison pill
                    break
                    
                # Play audio chunk
                sd.play(audio_data, self.samplerate)
                sd.wait()
                
                # Explicit cleanup
                del audio_data
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Playback error: {e}")
                
    def _process_text_stream(self, text_chunks, save_audio=False):
        """Generator that processes text chunks and yields audio"""
        chunk_index = 0
        
        for text_chunk in text_chunks:
            if self.stop_playback.is_set():
                break
                
            try:
                # Generate audio for this text chunk
                generator = self.pipeline(text_chunk, voice=self.voice)
                
                audio_index = 0
                for gs, ps, audio in generator:
                    if self.stop_playback.is_set():
                        break
                    
                    # Convert to numpy array if needed and ensure float32
                    if not isinstance(audio, np.ndarray):
                        audio = np.array(audio, dtype=np.float32)
                    else:
                        audio = audio.astype(np.float32)
                    
                    # Normalize audio to prevent clipping
                    if audio.max() > 1.0 or audio.min() < -1.0:
                        audio = audio / np.max(np.abs(audio))
                    
                    # Save audio file if requested
                    if save_audio:
                        sf.write(f'{chunk_index}_{audio_index}.wav', audio, self.samplerate)
                    
                    yield audio
                    
                    audio_index += 1
                    
                    # Periodic garbage collection
                    if audio_index % 10 == 0:
                        gc.collect()
                        
            except Exception as e:
                print(f"Processing error for chunk {chunk_index}: {e}")
                continue
                
            chunk_index += 1
    
    def _read_file_chunks(self, filename, chunk_size=500):
        """Memory-efficient file reading with sentence-aware chunking"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                buffer = ''
                sentence_endings = {'.', '!', '?', '\n'}
                
                for line in f:
                    buffer += line.strip() + ' '
                    
                    # Check if we have enough content and end on sentence boundary
                    if len(buffer) >= chunk_size:
                        # Find the last sentence ending
                        last_sentence_end = -1
                        for i in range(len(buffer) - 1, max(0, len(buffer) - 200), -1):
                            if buffer[i] in sentence_endings:
                                last_sentence_end = i
                                break
                        
                        if last_sentence_end > chunk_size // 2:  # Ensure meaningful chunk size
                            yield buffer[:last_sentence_end + 1].strip()
                            buffer = buffer[last_sentence_end + 1:].strip() + ' '
                        else:
                            # Fallback: split at chunk_size
                            yield buffer[:chunk_size].strip()
                            buffer = buffer[chunk_size:].strip() + ' '
                
                # Yield remaining buffer
                if buffer.strip():
                    yield buffer.strip()
                    
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    def start_streaming(self, filename, chunk_size=500, save_audio=False):
        """Start streaming audio from text file"""
        if self.is_playing:
            print("Already playing. Stop current stream first.")
            return
            
        self.stop_playback.clear()
        self.is_playing = True
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        # Start processing in separate thread
        def processing_worker():
            try:
                text_chunks = self._read_file_chunks(filename, chunk_size)
                audio_generator = self._process_text_stream(text_chunks, save_audio)
                
                for audio_chunk in audio_generator:
                    if self.stop_playback.is_set():
                        break
                        
                    # Add to playback queue (this will block if queue is full)
                    try:
                        self.audio_queue.put(audio_chunk, timeout=1.0)
                    except queue.Full:
                        print("Audio queue full, dropping chunk")
                        continue
                        
            except Exception as e:
                print(f"Processing worker error: {e}")
            finally:
                # Signal end of stream
                self.audio_queue.put(None)  # Poison pill
                self.is_playing = False
        
        processing_thread = threading.Thread(target=processing_worker, daemon=True)
        processing_thread.start()
        
        return processing_thread
    
    def stop_streaming(self):
        """Stop current streaming"""
        self.stop_playback.set()
        self.is_playing = False
        
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for playback thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
            
        # Force garbage collection
        gc.collect()
    
    def get_queue_size(self):
        """Get current audio queue size for monitoring"""
        return self.audio_queue.qsize()

# Usage example
def main():
    # Create streamer instance
    streamer = EfficientTTSStreamer(
        lang_code='a',
        voice='af_bella',
        buffer_size=3,  # Keep only 3 audio chunks in memory
        samplerate=24000
    )
    
    print("Starting streaming...")
    processing_thread = streamer.start_streaming(
        'pg600.txt', 
        chunk_size=300,  # Smaller chunks for lower latency
        save_audio=True  # Set to False to save disk space
    )
    
    # Monitor streaming (optional)
    try:
        while streamer.is_playing:
            print(f"Queue size: {streamer.get_queue_size()}")
            time.sleep(2)
            
        # Wait for processing to complete
        if processing_thread:
            processing_thread.join()
            
    except KeyboardInterrupt:
        print("\nStopping...")
        streamer.stop_streaming()
    
    print("Streaming completed")

if __name__ == "__main__":
    main()
