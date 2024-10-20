import asyncio
import base64
import json
import logging
import io
import wave
import os
import numpy as np
import websockets
import soundfile
import copy

from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.playback import play
from pyaudio import Stream as PyAudioStream
from concurrent.futures import ThreadPoolExecutor
from audio_processing import process_audio_chunk

from emotion_matcher import EmotionMatcher
from negotiation_manager import NegotiationManager
from scenario_manager import ScenarioManager
from llm_demand_checker import LLMDemandChecker

#custom file imports
from emotion_matcher import EmotionMatcher
from emotion_matcher import EmotionMatcher
from negotiation_manager import NegotiationManager
from scenario_manager import ScenarioManager
from llm_demand_checker import LLMDemandChecker

executor = ThreadPoolExecutor(max_workers=1)
# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG)

class Connection:
    load_dotenv()
    user_emotion_scores = []
    ai_emotion_scores = []
    emotion_matcher = EmotionMatcher()
    scenario_manager = ScenarioManager()
    llm_checker = LLMDemandChecker(os.getenv('GEMINI_API_KEY'))
    negotiation_manager = NegotiationManager(emotion_matcher, scenario_manager, llm_checker)
    current_context = None

    @classmethod
    def initialize_scenario(cls, scenario_name):
        cls.scenario_manager.select_scenario(scenario_name)
        cls.negotiation_manager.load_demands()
        cls._update_context()

    @classmethod
    def get_available_scenarios(cls):
        return cls.scenario_manager.get_scenario_names()

    @classmethod
    def _update_context(cls):
        try:
            current_demands = cls.negotiation_manager.get_current_demands()
            scenario = cls.scenario_manager.get_current_scenario()
            
            context = copy.deepcopy(scenario)
            for i, demand in enumerate(context["demands"]):
                current_value = current_demands[i][2]  # This is the current_value, not necessarily the level
                levels = demand.get("levels", [])
                
                if not levels:
                    logging.error(f"No levels found for demand {i}: {demand['description']}")
                    continue
                
                # Find the appropriate level based on the current_value
                current_level = 0
                for j, level in enumerate(levels):
                    if str(current_value) == str(level['value']):
                        current_level = j
                        break
                else:
                    logging.warning(f"No exact match found for current value {current_value} in demand {i}: {demand['description']}. Using the highest level.")
                    current_level = len(levels) - 1

                demand.update(levels[current_level])
                del demand["levels"]  # Remove the levels array from the context

            cls.current_context = {
                "text": json.dumps(context),
                "type": "editable"
            }
            logging.info("Context updated successfully")
        except Exception as e:
            logging.error(f"Error in _update_context: {e}")
            cls.current_context = {
                "text": json.dumps({"error": "Failed to update context"}),
                "type": "editable"
            }

    @classmethod
    async def connect(cls, socket_url, audio_stream, sample_rate, sample_width, num_channels, chunk_size, audio_processor=None):
        while True:
            try:
                logging.info("Attempting to connect to WebSocket...")
                async with websockets.connect(socket_url) as socket:
                    logging.info("Connected to WebSocket")
                    audio_queue = asyncio.Queue()
                    send_task = asyncio.create_task(
                        cls._send_audio_data(socket, audio_stream, sample_rate, sample_width, num_channels, chunk_size, audio_processor)
                    )
                    receive_task = asyncio.create_task(cls._receive_data(socket, audio_queue))
                    playback_task = asyncio.create_task(cls._play_audio_from_queue(audio_queue))
                    context_injection_task = asyncio.create_task(cls._inject_context(socket))
                    
                    logging.info("Starting send, receive, playback, and context injection tasks")
                    await asyncio.gather(receive_task, send_task, playback_task, context_injection_task)
            except websockets.exceptions.ConnectionClosed:
                logging.error("WebSocket connection closed. Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"An error occurred in connect: {e}. Attempting to reconnect in 5 seconds...")
                await asyncio.sleep(5)

    @classmethod
    async def _inject_context(cls, socket):
        while True:
            try:
                cls._update_context()  # Update the context before injection

                # Prepare context message
                context_message = {
                    "type": "session_settings",
                    "payload": {
                        "context": cls.current_context
                    }
                }

                # Additional dynamic information
                additional_info = {
                    "success_score": cls.emotion_matcher.get_success_score(),
                    "average_emotion_match": cls.emotion_matcher.get_average_match(),
                }

                # Merge additional info into the context
                context_data = json.loads(cls.current_context["text"])
                context_data.update(additional_info)
                cls.current_context["text"] = json.dumps(context_data)

                # Inject context
                await socket.send(json.dumps(context_message))
                logging.info("Context injected successfully")

                # Wait before next injection
                await asyncio.sleep(10)  # Adjust timing as needed
            except Exception as e:
                logging.error(f"Error in context injection: {e}")
                await asyncio.sleep(5)

    @classmethod
    async def _receive_data(cls, socket, audio_queue):
        try:
            logging.info("Starting receive loop")
            async for message in socket:
                try:
                    json_message = json.loads(message)
                    if json_message.get("type") == "audio_output":
                        audio_data = base64.b64decode(json_message["data"])
                        audio_length = len(audio_data)
                        logging.info(f"Received audio data, length: {audio_length} bytes")
                        await audio_queue.put(audio_data)
                    elif json_message.get("type") == "user_message":
                        cls._process_emotion_scores(json_message, is_user=True)
                    elif json_message.get("type") == "assistant_message":
                        cls._process_emotion_scores(json_message, is_user=False)
                    else:
                        logging.debug(f"Received non-audio message: {json_message}")
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON: {e}")
                except KeyError as e:
                    logging.error(f"Key error in JSON data: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error processing message: {e}")
        except Exception as e:
            logging.error(f"An error occurred in receive loop: {e}")
        finally:
            logging.info("Exiting receive loop")

    @classmethod
    async def update_scenario_during_conversation(cls, socket, demand_index, new_level):
        cls.negotiation_manager.adjust_demand(demand_index, new_level)
        cls._update_context()
        update_message = {
            "type": "session_settings",
            "payload": {
                "context": cls.current_context
            }
        }
        await socket.send(json.dumps(update_message))

    @classmethod
    def _process_emotion_scores(cls, json_message, is_user):
        try:
            prosody_scores = json_message.get("models", {}).get("prosody", {}).get("scores", {})
            if prosody_scores:
                sorted_scores = sorted(prosody_scores.items(), key=lambda x: x[1], reverse=True)
                top_5_scores = sorted_scores[:5]
                
                if is_user:
                    cls.user_emotion_scores.append(top_5_scores)
                    match_score, user_emotions, best_ai_emotions, success_increment, range_info = cls.emotion_matcher.add_user_emotion(top_5_scores)
                    print("Top 5 User Emotion Scores:")
                    for emotion, score in user_emotions:
                        print(f"{emotion}: {score:.3f}")
                    print()

                    user_message = json_message.get("text", "")
                    cls.negotiation_manager.update_transcript(f"User: {user_message}")
                    met_demands = cls.negotiation_manager.check_demands()
                    if met_demands:
                        print("\nDemands met in this message:")
                        for index, message in met_demands:
                            print(message)
                    
                    cls.negotiation_manager.adjust_demands()  # Adjust demands based on new success score
                    cls._update_context()  # Update context after demand adjustments
                    
                    if best_ai_emotions:
                        print("Best Matching AI Emotion Scores:")
                        for emotion, score in best_ai_emotions:
                            print(f"{emotion}: {score:.3f}")
                    print()
                    
                    print(f"Emotion match score: {match_score:.2f}")
                    print(f"Match range: {range_info}")
                    print(f"Success increment: {success_increment}")
                    print(f"Cumulative success score: {cls.emotion_matcher.get_success_score()}")
                    print(f"Average emotion match: {cls.emotion_matcher.get_average_match():.2f}")
                else:
                    cls.ai_emotion_scores.append(top_5_scores)
                    ai_emotions = cls.emotion_matcher.add_ai_emotions(top_5_scores)
                    print("Top 5 AI Emotion Scores:")
                    for emotion, score in ai_emotions:
                        print(f"{emotion}: {score:.3f}")
                    print()

                    ai_message = json_message.get("text", "")
                    cls.negotiation_manager.update_transcript(f"AI: {ai_message}")
                
                print()  # Add an empty line for better readability
            else:
                logging.warning(f"Prosody scores not found in the {'user' if is_user else 'AI'} message")
        except Exception as e:
            logging.error(f"Error processing {'user' if is_user else 'AI'} emotion scores: {e}")


    @classmethod
    def get_user_emotion_scores(cls):
        return cls.user_emotion_scores

    @classmethod
    def get_ai_emotion_scores(cls):
        return cls.ai_emotion_scores

    @classmethod
    async def _play_audio_from_queue(cls, audio_queue):
        while True:
            try:
                audio_data = await audio_queue.get()
                audio = AudioSegment.from_wav(io.BytesIO(audio_data))
                await asyncio.to_thread(play, audio)
                logging.info("Audio played successfully")
            except Exception as e:
                logging.error(f"Error playing audio: {e}")

    # The _send_audio_data and _read_audio_stream_non_blocking methods remain unchanged

    @classmethod
    async def _send_audio_data(cls, socket, audio_stream, sample_rate, sample_width, num_channels, chunk_size, audio_processor=None):
        wav_buffer = io.BytesIO()
        headers_sent = False

        try:
            logging.info("Starting send loop")
            while True:
                try:
                    data = await cls._read_audio_stream_non_blocking(audio_stream, chunk_size)
                    if num_channels == 2:
                        stereo_data = np.frombuffer(data, dtype=np.int16)
                        mono_data = ((stereo_data[0::2] + stereo_data[1::2]) / 2).astype(np.int16)
                        data = mono_data.tobytes()

                    # Apply audio processing if provided
                    if audio_processor:
                        processed_data = audio_processor(data, sample_rate, 1)  # Always use 1 channel after conversion
                    else:
                        processed_data = data

                    np_array = np.frombuffer(processed_data, dtype="int16")
                    soundfile.write(wav_buffer, np_array, samplerate=sample_rate, subtype="PCM_16", format="RAW")

                    wav_content = wav_buffer.getvalue()
                    if not headers_sent:
                        header_buffer = io.BytesIO()
                        with wave.open(header_buffer, "wb") as wf:
                            wf.setnchannels(1)  # Always use 1 channel after conversion
                            wf.setsampwidth(sample_width)
                            wf.setframerate(sample_rate)
                            wf.setnframes(chunk_size)
                            wf.writeframes(b"")

                        headers = header_buffer.getvalue()
                        wav_content = headers + wav_content
                        headers_sent = True

                    encoded_audio = base64.b64encode(wav_content).decode('utf-8')
                    json_message = json.dumps({"type": "audio_input", "data": encoded_audio})
                    await socket.send(json_message)
                    # logging.debug("Processed audio data sent successfully")

                    json_message = {
                        "type": "audio_input",
                        "data": encoded_audio,
                        "context": cls.current_context
                    }
                    await socket.send(json.dumps(json_message))

                    wav_buffer = io.BytesIO()
                except Exception as e:
                    logging.error(f"Error sending audio data: {e}")
                    break
        except Exception as e:
            logging.error(f"An error occurred in send loop: {e}")
        finally:
            logging.info("Exiting send loop")

    # ... (other methods remain unchanged)

    @classmethod
    async def _read_audio_stream_non_blocking(cls, audio_stream, chunk_size):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, audio_stream.read, chunk_size, False)
        return data