#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Movie Explorer Example.

This example demonstrates how to create a conversational movie exploration bot using:
- TMDB API for real movie data (including cast information)
- Pipecat Flows for conversation management
- Node functions for API calls (get_movies, get_movie_details, get_similar_movies)
- Edge functions for state transitions (explore_movie, greeting, end)

The flow allows users to:
1. See what movies are currently playing or coming soon
2. Get detailed information about specific movies (including cast)
3. Find similar movies as recommendations

Requirements:
- TMDB API key (https://www.themoviedb.org/documentation/api)
- Daily room URL
- Google API key (also, pip install pipecat-ai[google])
- Deepgram API key
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Literal, TypedDict, Union

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

sys.path.append(str(Path(__file__).parent.parent))
from runner import configure

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
    cast: List[str]


class MoviesResult(FlowResult):
    movies: List[MovieBasic]


class MovieDetailsResult(FlowResult, MovieDetails):
    pass


class SimilarMoviesResult(FlowResult):
    movies: List[MovieBasic]


class ErrorResult(FlowResult):
    status: Literal["error"]
    error: str


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

    async def fetch_upcoming_movies(self, session: aiohttp.ClientSession) -> List[MovieBasic]:
        """Fetch upcoming movies from TMDB.

        Returns top 5 upcoming movies with basic information.
        """
        url = f"{self.base_url}/movie/upcoming"
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
async def _get_movies() -> Union[MoviesResult, ErrorResult]:
    """Helper for fetching current movies."""
    logger.debug("Calling TMDB API: get_movies")
    async with aiohttp.ClientSession() as session:
        try:
            movies = await tmdb_api.fetch_current_movies(session)
            logger.debug(f"TMDB API Response: {movies}")
            return MoviesResult(movies=movies)
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            return ErrorResult(status="error", error="Failed to fetch movies")


async def get_movies_then_explore() -> tuple[Union[MoviesResult, ErrorResult], str]:
    """Handler for fetching currenting movies then transitioning to the explore node if there are no errors."""
    result = await _get_movies()
    if result.get("status") == "error":
        return result, None
    return result, "explore_movie"


async def get_movies() -> tuple[Union[MoviesResult, ErrorResult], None]:
    """Handler for fetching current movies."""
    return (await _get_movies(), None)


async def _get_upcoming_movies() -> Union[MoviesResult, ErrorResult]:
    """Helper for fetching upcoming movies."""
    logger.debug("Calling TMDB API: get_upcoming_movies")
    async with aiohttp.ClientSession() as session:
        try:
            movies = await tmdb_api.fetch_upcoming_movies(session)
            logger.debug(f"TMDB API Response: {movies}")
            return MoviesResult(movies=movies)
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            return ErrorResult(status="error", error="Failed to fetch upcoming movies")


async def get_upcoming_movies_then_explore() -> tuple[Union[MoviesResult, ErrorResult], str]:
    """Handler for fetching upcoming movies and transitioning to the explore node."""
    result = await _get_upcoming_movies()
    if result.get("status") == "error":
        return result, None
    return result, "explore_movie"


async def get_upcoming_movies() -> tuple[Union[MoviesResult, ErrorResult], None]:
    """Handler for fetching upcoming movies."""
    return (await _get_upcoming_movies(), None)


async def get_movie_details(args: FlowArgs) -> tuple[Union[MovieDetailsResult, ErrorResult], None]:
    """Handler for fetching movie details including cast."""
    movie_id = args["movie_id"]
    logger.debug(f"Calling TMDB API: get_movie_details for ID {movie_id}")
    async with aiohttp.ClientSession() as session:
        try:
            details = await tmdb_api.fetch_movie_details(session, movie_id)
            logger.debug(f"TMDB API Response: {details}")
            result = MovieDetailsResult(**details)
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            result = ErrorResult(
                status="error", error=f"Failed to fetch details for movie {movie_id}"
            )
        return result, None


async def get_similar_movies(
    args: FlowArgs,
) -> tuple[Union[SimilarMoviesResult, ErrorResult], None]:
    """Handler for fetching similar movies."""
    movie_id = args["movie_id"]
    logger.debug(f"Calling TMDB API: get_similar_movies for ID {movie_id}")
    async with aiohttp.ClientSession() as session:
        try:
            similar = await tmdb_api.fetch_similar_movies(session, movie_id)
            logger.debug(f"TMDB API Response: {similar}")
            result = SimilarMoviesResult(movies=similar)
        except Exception as e:
            logger.error(f"TMDB API Error: {e}")
            result = ErrorResult(
                status="error", error=f"Failed to fetch similar movies for {movie_id}"
            )
        return result, None


async def end_conversation() -> tuple[None, str]:
    """Handler to end the conversation."""
    return None, "end"


# Flow configuration
flow_config: FlowConfig = {
    "initial_node": "greeting",
    "nodes": {
        "greeting": {
            "role_messages": [
                {
                    "role": "system",
                    "content": "You are a friendly movie expert. Your responses will be converted to audio, so avoid special characters. Always use the available functions to progress the conversation naturally.",
                }
            ],
            "task_messages": [
                {
                    "role": "system",
                    "content": "Start by greeting the user and asking if they'd like to know about movies currently in theaters or upcoming releases. Wait for their choice before using either get_current_movies_then_explore or get_upcoming_movies_then_explore.",
                }
            ],
            "functions": [
                {
                    "function_declarations": [
                        {
                            "name": "get_current_movies_then_explore",
                            "handler": get_movies_then_explore,
                            "description": "Fetch movies currently playing in theaters",
                            "parameters": None,  # Specify None for no parameters
                        },
                        {
                            "name": "get_upcoming_movies_then_explore",
                            "handler": get_upcoming_movies_then_explore,
                            "description": "Fetch movies coming soon to theaters",
                            "parameters": None,  # Specify None for no parameters
                        },
                    ]
                }
            ],
        },
        "explore_movie": {
            "task_messages": [
                {
                    "role": "user",
                    "content": """Help the user learn more about movies. You can:
- Use get_movie_details when they express interest in a specific movie
- Use get_similar_movies to show recommendations
- Use get_current_movies to see what's playing now
- Use get_upcoming_movies to see what's coming soon
- Use end_conversation when they're done exploring

After showing details or recommendations, ask if they'd like to explore another movie or end the conversation.""",
                }
            ],
            "functions": [
                {
                    "function_declarations": [
                        {
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
                        {
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
                        {
                            "name": "get_current_movies",
                            "handler": get_movies,
                            "description": "Show current movies in theaters",
                            "parameters": None,  # Specify None for no parameters
                        },
                        {
                            "name": "get_upcoming_movies",
                            "handler": get_upcoming_movies,
                            "description": "Show movies coming soon",
                            "parameters": None,  # Specify None for no parameters,
                        },
                        {
                            "name": "end_conversation",
                            "handler": end_conversation,
                            "description": "End the conversation",
                            "parameters": None,  # Specify None for no parameters,
                        },
                    ]
                }
            ],
        },
        "end": {
            "task_messages": [
                {
                    "role": "user",
                    "content": "Thank the user warmly and mention they can return anytime to discover more movies.",
                }
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
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="c45bc5ec-dc68-4feb-8829-6e6b2748095d",  # Movieman
            text_filter=MarkdownTextFilter(),
        )
        llm = GoogleLLMService(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-2.0-flash-exp")

        context = OpenAILLMContext()
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

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            flow_config=flow_config,
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await flow_manager.initialize()

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
