#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Utility functions for Pipecat Flows examples.

This module provides helper functions to reduce boilerplate and keep examples
focused on the core flow concepts.
"""

import os
from typing import Any


def create_llm() -> Any:
    """Create an LLM service instance based on environment configuration.

    Returns:
        Configured LLM service instance

    Raises:
        ValueError: If provider is unsupported or required API keys are missing

    Supported Providers:
        - openai: Requires OPENAI_API_KEY
        - anthropic: Requires ANTHROPIC_API_KEY
        - google: Requires GOOGLE_API_KEY
        - aws: Uses AWS default credential chain (SSO, environment variables, or IAM roles)
              Optionally set AWS_REGION (defaults to us-west-2)
        - gemini_live: Requires GOOGLE_API_KEY

    Usage:
        # Use provider from LLM_PROVIDER env var (defaults to OpenAI)
        llm = create_llm()
    """
    provider = _resolve_provider()

    # Provider configurations
    configs = {
        "openai": {
            "service": "pipecat.services.openai.llm.OpenAILLMService",
            "api_key_env": "OPENAI_API_KEY",
            "default_model": "gpt-4.1",
        },
        "anthropic": {
            "service": "pipecat.services.anthropic.llm.AnthropicLLMService",
            "api_key_env": "ANTHROPIC_API_KEY",
            "default_model": "claude-sonnet-4-5-20250929",
        },
        "google": {
            "service": "pipecat.services.google.llm.GoogleLLMService",
            "api_key_env": "GOOGLE_API_KEY",
            "default_model": "gemini-2.5-flash",
        },
        "aws": {
            "service": "pipecat.services.aws.llm.AWSBedrockLLMService",
            "api_key_env": None,  # AWS uses default credential chain
            "default_model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "region": "us-west-2",
        },
        "gemini_live": {
            "service": "pipecat.services.google.gemini_live.llm.GeminiLiveLLMService",
            "api_key_env": "GOOGLE_API_KEY",
            "default_model": "models/gemini-2.5-flash-native-audio-preview-12-2025",
        },
    }

    config = configs.get(provider)
    if not config:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unsupported LLM provider: {provider}. Available: {available}")

    # Dynamic import of the LLM service
    module_path, class_name = config["service"].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    service_class = getattr(module, class_name)

    # Get API key (skip for AWS which uses default credential chain)
    if provider == "aws" or config["api_key_env"] is None:
        api_key = None  # AWS uses default credential chain
    else:
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Missing API key: {config['api_key_env']} for provider: {provider}")

    # Use default model
    selected_model = config["default_model"]

    # Build kwargs
    kwargs = {"api_key": api_key, "model": selected_model}

    # Add AWS-specific parameters
    if provider == "aws":
        kwargs["aws_region"] = os.getenv("AWS_REGION", config["region"])
        kwargs["params"] = service_class.InputParams(temperature=0.8)
        # Remove the generic api_key since AWS uses default credential chain
        del kwargs["api_key"]

    return service_class(**kwargs)


def needs_stt_tts() -> bool:
    """Return True when STT/TTS should be enabled for the LLM service specified by environment configuration.

    Returns:
        True if STT/TTS should be enabled, False if the provider handles audio natively.
    """
    return _resolve_provider() != "gemini_live"


def _resolve_provider() -> str:
    """Resolve the LLM provider name from environment configuration.

    Returns:
        Lower-cased provider name.
    """
    return os.getenv("LLM_PROVIDER", "openai").lower()
