import sounddevice as sd
import threading
from kokoro import KPipeline
import soundfile as sf

pipeline = KPipeline(lang_code='a')

def play_audio(audio, samplerate=24000):
    sd.play(audio, samplerate)
    sd.wait()

def process_file_in_chunks(filename, chunk_size=1000):
    with open(filename, 'r') as f:
        buffer = ''
        for line in f:
            buffer += line.strip() + ' '
            if len(buffer) >= chunk_size:
                yield buffer.strip()
                buffer = ''
        if buffer:
            yield buffer.strip()

for chunk_index, text_chunk in enumerate(process_file_in_chunks('pg600.txt')):
    generator = pipeline(text_chunk, voice='af_bella')
    for i, (gs, ps, audio) in enumerate(generator):
        threading.Thread(target=play_audio, args=(audio,)).start()
        sf.write(f'{chunk_index}_{i}.wav', audio, 24000)

