import os
import asyncio
import logging
import signal
import sys
import pyaudio
import serial
import struct
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeadlessService")

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-native-audio-preview-12-2025"
BUTTON_PIN = 17 

# Audio Config
FORMAT = pyaudio.paInt16
CHANNELS_IN = 1
CHANNELS_OUT = 1
RATE_IN = 16000  # Gemini expects 16kHz for input, as mentioned in SDK
RATE_OUT = 24000 # Gemini outputs 24kHz
CHUNK = 512

# Visualizer / Serial Config
SERIAL_PORT = '/dev/ttyHS1'
BAUD = 115200
NUM_BARS = 13
MIN_FREQ = 40
MAX_FREQ = 12000
SMOOTHING = 0.5

if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables.")
    sys.exit(1)

# GPIO Setup
try:
    from gpiozero import Button
    HAS_GPIO = True
    logger.info("GPIO configuration loaded.")
except ImportError:
    HAS_GPIO = False
    logger.warning("gpiozero not found. Running in simulation mode (Button disabled).")
except Exception as e:
    HAS_GPIO = False
    logger.warning(f"GPIO initialization failed: {e}")

class VoiceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
        self.audio = pyaudio.PyAudio()
        self.session = None
        self.running = False
        self.input_stream = None
        self.output_stream = None
        self.button = None
        self.loop = None # Will be set in main
        
        # Visualizer State
        self.ser = None
        self.last_bars = np.zeros(NUM_BARS)
        self.max_val_seen = 1.0
        
        # Initialize Serial for Visualizer
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0.1)
            logger.info(f"Connected to Arduino on {SERIAL_PORT}")
        except Exception as e:
            logger.error(f"Serial Error (Visualizer disabled): {e}")

        if HAS_GPIO:
            try:
                self.button = Button(BUTTON_PIN, hold_time=2)
                self.button.when_pressed = self.toggle_session_callback
                logger.info(f"Button configured on GPIO {BUTTON_PIN}")
            except Exception as e:
                logger.error(f"Failed to configure button: {e}")

    def toggle_session_callback(self):
        """Callback wrapper to schedule async toggle safely from GPIO thread"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.toggle_session(), self.loop)

    async def toggle_session(self):
        if self.running:
            logger.info("Button pressed: Stopping session...")
            await self.stop_session()
        else:
            logger.info("Button pressed: Starting session...")
            await self.start_session()

    async def start_session(self):
        if self.running:
            return

        self.running = True
        logger.info("Initializing audio streams...")
        
        try:
            self.input_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS_IN,
                rate=RATE_IN,
                input=True,
                frames_per_buffer=CHUNK,
            )
            
            self.output_stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS_OUT,
                rate=RATE_OUT,
                output=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            logger.error(f"Failed to open audio streams: {e}")
            self.running = False
            return

        logger.info("Connecting to Gemini...")
        
        config = {
            "response_modalities": ["AUDIO"],
            "speech_config": {
                "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
            },
            "system_instruction": """
                You are 'Commbox Assistant', the world's most advanced cricket analysis engine.
                CRITICAL: You MUST respond in the EXACT SAME language that the user speaks to you in.
                Keep responses punchy, conversational, and voice-optimized.
            """
        }

        try:
            async with self.client.aio.live.connect(model=MODEL_NAME, config=config) as session:
                self.session = session
                logger.info("Gemini Live session established.")
                
                await asyncio.gather(
                    self.send_audio_loop(),
                    self.receive_audio_loop()
                )
        except Exception as e:
            logger.error(f"Session error: {e}")
        finally:
            await self.stop_session()

    async def stop_session(self):
        if not self.running:
            return
            
        logger.info("Closing session resources...")
        self.running = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
            
        logger.info("Session stopped. Waiting for button press.")

    def process_visualizer(self, audio_data):
        """Calculates FFT and sends data to Arduino. No asyncio wait needed here."""
        if not self.ser or not self.ser.is_open:
            return

        # 1. Convert PCM bytes to Numpy Float (-1.0 to 1.0)
        # Gemini sends Int16 bytes
        indata = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize Int16 to Float -1..1
        indata = indata / 32768.0

        # 2. FFT Logic (Adapted from your script)
        # Note: We use RATE_OUT (24000) instead of 44100
        if len(indata) == 0: return

        window = np.hanning(len(indata))
        fft_vals = np.abs(np.fft.rfft(indata * window))
        freqs = np.fft.rfftfreq(len(indata), 1.0/RATE_OUT)
        
        edges = np.geomspace(MIN_FREQ, MAX_FREQ, NUM_BARS + 1)
        current_bars = []
        
        for i in range(NUM_BARS):
            mask = (freqs >= edges[i]) & (freqs < edges[i+1])
            if np.any(mask):
                raw_val = np.mean(fft_vals[mask])
            else:
                raw_val = 0.0
            current_bars.append(raw_val)

        # 3. Dynamic Gain
        self.max_val_seen *= 0.995
        curr_max = max(current_bars) if current_bars else 0
        if curr_max > self.max_val_seen:
            self.max_val_seen = curr_max
            
        if self.max_val_seen < 0.0001: self.max_val_seen = 0.0001
            
        # 4. Scale and Clamp
        final_bars = []
        for val in current_bars:
            norm = val / self.max_val_seen
            scaled = np.sqrt(norm) * 8.0
            final_bars.append(scaled)
            
        # 5. Smoothing
        self.last_bars = (self.last_bars * SMOOTHING) + (np.array(final_bars) * (1.0 - SMOOTHING))
        
        # 6. Send to Arduino
        byte_data = [int(min(max(x, 0), 8)) for x in self.last_bars]
        try:
            self.ser.write(b'V' + struct.pack(f'{NUM_BARS}B', *byte_data))
        except Exception:
            pass # Don't crash main thread on serial glitch

    async def send_audio_loop(self):
        logger.info("Started microphone input loop")
        try:
            while self.running:
                data = await self.loop.run_in_executor(
                    None, 
                    lambda: self.input_stream.read(CHUNK, exception_on_overflow=False)
                )
                
                if self.session:
                    await self.session.send(
                        input={"data": data, "mime_type": "audio/pcm"}, 
                        end_of_turn=False
                    )
        except Exception as e:
            logger.error(f"Error in send_audio_loop: {e}")
            self.running = False

    async def receive_audio_loop(self):
        logger.info("Started audio output loop")
        try:
            while self.running:
                async for response in self.session.receive():
                    if not self.running: break
                        
                    server_content = response.server_content
                    if server_content is None: continue

                    model_turn = server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.inline_data:
                                audio_bytes = part.inline_data.data
                                
                                # Task 1: Play Audio (Blocking via Executor)
                                await self.loop.run_in_executor(
                                    None,
                                    lambda: self.output_stream.write(audio_bytes)
                                )
                                
                                # Task 2: Update Visualizer (Fire and forget via Executor)
                                # We run this in executor so math doesn't block the next audio chunk
                                self.loop.run_in_executor(
                                    None,
                                    lambda: self.process_visualizer(audio_bytes)
                                )
                                
        except Exception as e:
            logger.error(f"Error in receive_audio_loop: {e}")
        finally:
            self.running = False

async def main():
    agent = VoiceAgent()
    agent.loop = asyncio.get_running_loop()
    
    logger.info("Service started. Starting session automatically...")
    asyncio.create_task(agent.start_session())
    
    stop_event = asyncio.Event()
    
    def signal_handler():
        logger.info("Shutting down...")
        stop_event.set()
        
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        if agent.running:
            await agent.stop_session()
        if agent.ser and agent.ser.is_open:
            agent.ser.close()
        agent.audio.terminate()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

