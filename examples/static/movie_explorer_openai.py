#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
# Movie Explorer Example
#
# This example demonstrates how to create a conversational movie exploration bot using:
# - TMDB API for real movie data (including cast information)
# - Pipecat Flows for conversation management
# - Node functions for API calls (get_movies, get_movie_details, get_similar_movies)
# - Edge functions for state transitions (explore_movie, greeting, end)
#
# The flow allows users to:
# 1. See what movies are currently playing
# 2. Get detailed information about specific movies (including cast)
# 3. Find similar movies as recommendations
#
# Requirements:
# - TMDB API key (https://www.themoviedb.org/documentation/api)
# - Daily room URL
# - OpenAI API key
# - Deepgram API key

import asyncio
import os
import sys
from pathlib import Path
from typing import List, TypedDict

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# TMDB API setup
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"


# Type definitions for API responses
class MovieBasic(TypedDict):
    id: int
    title: str
    overview: str


class MovieDetails(TypedDict):
    title: str
    runtime: int
    rating: float
    overview: str
    genres: List[str]
    cast: List[str]  # List of "Actor Name as Character Name"


class TMDBApi:
    """Handles all TMDB API interactions with proper typing and error handling."""

    def __init__(self, api_key: str, base_url: str = "https://api.themoviedb.org/3"):
        self.api_key = api_key
        self.base_url = base_url

    async def fetch_current_movies(self, session: aiohttp.ClientSession) -> List[MovieBasic]:
        """Fetch currently playing movies from TMDB.

        Returns top 5 movies with basic information.
        """
        url = f"{self.base_url}/movie/now_playing"
        params = {"api_key": self.api_key, "language": "en-US", "page": 1}

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"TMDB API Error: {response.status}")
                raise ValueError(f"API returned status {response.status}")

            data = await response.json()
            if "results" not in data:
                logger.error(f"Unexpected API response: {data}")
                raise ValueError("Invalid API response format")

            return [
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "overview": movie["overview"][:100] + "...",
                }
                for movie in data["results"][:5]
            ]

    async def fetch_movie_credits(self, session: aiohttp.ClientSession, movie_id: int) -> List[str]:
        """Fetch top cast members for a movie.

        Returns list of strings in format: "Actor Name as Character Name"
        """
        url = f"{self.base_url}/movie/{movie_id}/credits"
        params = {"api_key": self.api_key, "language": "en-US"}

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"TMDB API Error: {response.status}")
                raise ValueError(f"API returned status {response.status}")

            data = await response.json()
            if "cast" not in data:
                logger.error(f"Unexpected API response: {data}")
                raise ValueError("Invalid API response format")

            return [
                f"{actor['name']} as {actor['character']}"
                for actor in data["cast"][:5]  # Top 5 cast members
            ]

    async def fetch_movie_details(
        self, session: aiohttp.ClientSession, movie_id: int
    ) -> MovieDetails:
        """Fetch detailed information about a specific movie, including cast."""
        # Fetch basic movie details
        url = f"{self.base_url}/movie/{movie_id}"
        params = {"api_key": self.api_key, "language": "en-US"}

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"TMDB API Error: {response.status}")
                raise ValueError(f"API returned status {response.status}")

            data = await response.json()
            required_fields = ["title", "runtime", "vote_average", "overview", "genres"]
            if not all(field in data for field in required_fields):
                logger.error(f"Missing required fields in response: {data}")
                raise ValueError("Invalid API response format")

            # Fetch cast information
            cast = await self.fetch_movie_credits(session, movie_id)

            return {
                "title": data["title"],
                "runtime": data["runtime"],
                "rating": data["vote_average"],
                "overview": data["overview"],
                "genres": [genre["name"] for genre in data["genres"]],
                "cast": cast,
            }

    async def fetch_similar_movies(
        self, session: aiohttp.ClientSession, movie_id: int
    ) -> List[MovieBasic]:
        """Fetch movies similar to the specified movie.

        Returns top 3 similar movies with basic information.
        """
        url = f"{self.base_url}/movie/{movie_id}/similar"
        params = {"api_key": self.api_key, "language": "en-US", "page": 1}

        async with session.get(url, params=params) as response:
            if response.status != 200:
                logger.error(f"TMDB API Error: {response.status}")
                raise ValueError(f"API returned status {response.status}")

            data = await response.json()
            if "results" not in data:
                logger.error(f"Unexpected API response: {data}")
                raise ValueError("Invalid API response format")

            return [
                {
                    "id": movie["id"],
                    "title": movie["title"],
                    "overview": movie["overview"][:100] + "...",
                }
                for movie in data["results"][:3]
            ]


# Create TMDB API instance
tmdb_api = TMDBApi(TMDB_API_KEY)


# Function handlers for the LLM
# These are node functions that perform operations without changing conversation state
async def get_movies() -> FlowResult:
    """Handler for fetching current movies."""
    logger.debug("Calling TMDB API: get_movies")
    async with aiohttp.ClientSession() as session:
        try:
            movies = await tmdb_api.fetch_current_movies(session)
            logger.debug(f"TMDB API Response: {movies}")
            return {"movies": movies}
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            return {"error": "Failed to fetch movies"}


async def get_movie_details(args: FlowArgs) -> FlowResult:
    """Handler for fetching movie details including cast."""
    movie_id = args["movie_id"]
    logger.debug(f"Calling TMDB API: get_movie_details for ID {movie_id}")
    async with aiohttp.ClientSession() as session:
        try:
            details = await tmdb_api.fetch_movie_details(session, movie_id)
            logger.debug(f"TMDB API Response: {details}")
            return details
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            return {"error": f"Failed to fetch details for movie {movie_id}"}


async def get_similar_movies(args: FlowArgs) -> FlowResult:
    """Handler for fetching similar movies."""
    movie_id = args["movie_id"]
    logger.debug(f"Calling TMDB API: get_similar_movies for ID {movie_id}")
    async with aiohttp.ClientSession() as session:
        try:
            similar = await tmdb_api.fetch_similar_movies(session, movie_id)
            logger.debug(f"TMDB API Response: {similar}")
            return {"movies": similar}
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            return {"error": f"Failed to fetch similar movies for {movie_id}"}


# Flow configuration
flow_config: FlowConfig = {
    "initial_node": "greeting",
    "nodes": {
        "greeting": {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful movie expert. Start by greeting the user and asking if they'd like to know what movies are currently playing in theaters. Use get_movies to fetch the current movies when they're interested.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_movies",
                        "handler": get_movies,
                        "description": "Fetch currently playing movies",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "explore_movie",
                        "description": "Move to movie exploration",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "pre_actions": [
                {
                    "type": "tts_say",
                    "text": "Welcome! I can tell you about movies currently playing in theaters.",
                }
            ],
        },
        "explore_movie": {
            "messages": [
                {
                    "role": "system",
                    "content": "Help the user learn more about movies. Use get_movie_details when they express interest in a specific movie - this will show details including cast, runtime, and rating. After showing details, you can use get_similar_movies if they want recommendations. Ask if they'd like to explore another movie (use explore_movie) or end the conversation.",
                }
            ],
            "functions": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_movie_details",
                        "handler": get_movie_details,
                        "description": "Get details about a specific movie including cast",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "movie_id": {"type": "integer", "description": "TMDB movie ID"}
                            },
                            "required": ["movie_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_similar_movies",
                        "handler": get_similar_movies,
                        "description": "Get similar movies as recommendations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "movie_id": {"type": "integer", "description": "TMDB movie ID"}
                            },
                            "required": ["movie_id"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "explore_movie",
                        "description": "Return to current movies list",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "end",
                        "description": "End the conversation",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        },
        "end": {
            "messages": [{"role": "system", "content": "Thank the user and end the conversation."}],
            "functions": [],
            "pre_actions": [
                {"type": "tts_say", "text": "Thanks for exploring movies with me! Goodbye!"}
            ],
            "post_actions": [{"type": "end_conversation"}],
        },
    },
}


async def main():
    """Main function to set up and run the movie explorer bot."""
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)

        transport = DailyTransport(
            room_url,
            None,
            "Movie Explorer Bot",
            DailyParams(
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = DeepgramTTSService(api_key=os.getenv("DEEPGRAM_API_KEY"), voice="aura-helios-en")
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        # Get initial tools from the first node
        initial_tools = flow_config["nodes"]["greeting"]["functions"]

        # Create initial context
        messages = [
            {
                "role": "system",
                "content": "You are a friendly movie expert. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.",
            }
        ]

        context = OpenAILLMContext(messages, initial_tools)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(task, llm, tts, flow_config)

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await flow_manager.initialize(messages)
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
