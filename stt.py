import asyncio
import signal
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.logging import configure_pretty_logging
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.models.transcriber import (DeepgramTranscriberConfig,
                                                 PunctuationEndpointingConfig,
                                                 AzureTranscriberConfig
                                                 )
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber
from vocode.streaming.transcriber.azure_transcriber import AzureTranscriber

configure_pretty_logging()

# Make sure to load environment variables
load_dotenv()

class Settings(BaseSettings):
    """
    Settings for the streaming conversation quickstart.
    These parameters can be configured with environment variables.
    """

    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    azure_speech_key: str = os.getenv("AZURE_SPEECH_KEY")
    deepgram_api_key: str = os.getenv("DEEPGRAM_API_KEY")
    azure_speech_region: str = "eastus"

    # This means a .env file can be used to overload these settings
    # ex: "OPENAI_API_KEY=my_key" will set openai_api_key over the default above
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


async def main():
    (microphone_input, speaker_output,) = create_streaming_microphone_input_and_speaker_output(use_default_devices=False,)
    conversation = StreamingConversation(
        output_device=speaker_output,
        # transcriber=DeepgramTranscriber(
        #     DeepgramTranscriberConfig.from_input_device(
        #         microphone_input,
        #         endpointing_config=PunctuationEndpointingConfig(),
        #         api_key=settings.deepgram_api_key,
        #     ),
        # ),
        transcriber=AzureTranscriber(
            azure_speech_key=os.getenv("AZURE_SPEECH_KEY"),
            azure_speech_region=os.getenv("AZURE_SPEECH_REGION"),
            endpointing_config=PunctuationEndpointingConfig(),
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                openai_api_key=settings.openai_api_key,
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""You a math expert, guide any questions about math""",
            )
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output),
            azure_speech_key=settings.azure_speech_key,
            azure_speech_region=settings.azure_speech_region,
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate()))
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())