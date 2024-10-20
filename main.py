# main.py

import asyncio
import os
import logging
from authenticator import Authenticator
from connection import Connection
from devices import AudioDevices
from dotenv import load_dotenv
from pyaudio import PyAudio, paInt16

from audio_processing import process_audio_chunk  # Add this import
from dotenv import load_dotenv
load_dotenv()

# Audio format and parameters
FORMAT = paInt16
CHANNELS = 1
SAMPLE_WIDTH = 2  # PyAudio.get_sample_size(pyaudio, format=paInt16)
CHUNK_SIZE = 1024

def select_configuration():
    load_dotenv()  # Load environment variables from .env file

    configs = {
        1: ("David", os.getenv("HUME_CONFIG_ID_1")),
        2: ("Ashley", os.getenv("HUME_CONFIG_ID_2")),
        3: ("Jessica", os.getenv("HUME_CONFIG_ID_3"))
    }

    print("Available configurations:")
    for num, (name, config_id) in configs.items():
        print(f"{num}: {name} (Config ID: {config_id})")

    while True:
        try:
            choice = int(input("Select a configuration (1-3): "))
            if choice in configs:
                return configs[choice]
            else:
                print("Invalid choice. Please select a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_scenario_and_configuration():
    scenarios = Connection.get_available_scenarios()
    
    print("Available scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")
    
    while True:
        try:
            scenario_choice = int(input("Select a scenario (enter the number): "))
            if 1 <= scenario_choice <= len(scenarios):
                selected_scenario = scenarios[scenario_choice - 1]
                break
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print(f"\nSelected scenario: {selected_scenario}")
    print("\nNow, select a configuration for the AI personality:")
    
    config_name, config_id = select_configuration()
    
    return selected_scenario, config_name, config_id


async def main():
    pyaudio_instance = PyAudio()

    selected_scenario, config_name, selected_config = select_scenario_and_configuration()
    print(f"Selected scenario: {selected_scenario}")
    print(f"Selected configuration: {config_name} (Config ID: {selected_config})")
    
    Connection.initialize_scenario(selected_scenario)
    
    try:
        input_devices, output_devices = AudioDevices.list_audio_devices(pyaudio_instance)
        input_device_index, input_device_sample_rate = AudioDevices.choose_device(input_devices, "input")
        output_device_index = AudioDevices.choose_device(output_devices, "output")

        HUME_CONFIG_ID = selected_config

        access_token = get_access_token()
        # socket_url = f"wss://api.hume.ai/v0/assistant/chat?access_token={access_token}"
        socket_url = f"wss://api.hume.ai/v0/evi/chat?access_token={access_token}&config_id={HUME_CONFIG_ID}"

        while True:
            try:
                audio_stream = pyaudio_instance.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    frames_per_buffer=CHUNK_SIZE,
                    rate=input_device_sample_rate,
                    input=True,
                    output=True,
                    input_device_index=input_device_index,
                    output_device_index=output_device_index,
                )

                logging.info("Starting Connection.connect")
                await Connection.connect(
                    socket_url,
                    audio_stream,
                    input_device_sample_rate,
                    SAMPLE_WIDTH,
                    CHANNELS,
                    CHUNK_SIZE,
                    process_audio_chunk  # Pass the audio processing function as an optional argument
                )

                # get the user_message.models.prosody.scores 
                # ^ json object with the scores for each emotion 
                # sort the scores by highest value first 
                # take the top 3 highest scores 
                # print them out 
                
            except Exception as e:
                logging.error(f"An error occurred in main loop: {e}")
                if 'audio_stream' in locals() and audio_stream.is_active():
                    audio_stream.stop_stream()
                    audio_stream.close()
                await asyncio.sleep(5)  # Wait before retrying
    except KeyboardInterrupt:
        logging.info("Program terminated by user")
    finally:
        if 'audio_stream' in locals() and audio_stream.is_active():
            audio_stream.stop_stream()
            audio_stream.close()
        pyaudio_instance.terminate()
        logging.info("PyAudio terminated")


def get_access_token() -> str:
    """
    Load API credentials from environment variables and fetch an access token.

    Returns:
        str: The access token.

    Raises:
        SystemExit: If API key or Secret key are not set.
    """
    load_dotenv()

    # Attempt to retrieve API key and Secret key from environment variables
    HUME_API_KEY = os.getenv("HUME_API_KEY")
    HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")

    # Ensure API key and Secret key are set
    if HUME_API_KEY is None or HUME_SECRET_KEY is None:
        print(
            "Error: HUME_API_KEY and HUME_SECRET_KEY must be set either in a .env file or as environment variables."
        )
        exit()

    # Create an instance of Authenticator with the API key and Secret key
    authenticator = Authenticator(HUME_API_KEY, HUME_SECRET_KEY)

    # Fetch the access token
    access_token = authenticator.fetch_access_token()
    return access_token


if __name__ == "__main__":
    """
    Entry point for the script. Runs the main asynchronous function.
    """
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(main())
