# apps/backend/app/agent/providers/groq.py

import os
import logging
from typing import Any, Dict
from fastapi.concurrency import run_in_threadpool
from groq import Groq

from ..exceptions import ProviderError
from .base import Provider, EmbeddingProvider
from ...core import settings

logger = logging.getLogger(__name__)


class GroqLLMProvider(Provider):
    """
    Handles interactions with Groq's LLM API.
    """

    def __init__(self, api_key: str | None = None, model_name: str = settings.GROQ_MODEL_NAME, opts: Dict[str, Any] = None):
        if opts is None:
            opts = {}
        api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderError("Groq API key is missing")
        
        try:
            self._client = Groq(api_key=api_key)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {e}") from e

        self.model_name = model_name
        self.opts = opts
        self.instructions = ""  # Optional instructions for the model
        logger.info(f"GroqLLMProvider initialized with model: {self.model_name}")

    def _generate_sync(self, prompt: str, options: Dict[str, Any]) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            chat_completion = self._client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=options.get("temperature", 0.7)
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling Groq API: {e}")
            raise ProviderError(f"Groq - error generating response: {e}") from e

    async def __call__(self, prompt: str, **generation_args: Any) -> str:
        myopts = {
            "temperature": self.opts.get("temperature", 0.7),
            "top_p": self.opts.get("top_p", 0.9),
        }
        # Run the blocking call in a threadpool for async safety
        return await run_in_threadpool(self._generate_sync, prompt, myopts)


class GroqEmbeddingProvider(EmbeddingProvider):
    """
    Generates embeddings using Groq LLMs (if supported).
    """

    def __init__(self, api_key: str | None = None, embedding_model: str = settings.EMBEDDING_MODEL):
        api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderError("Groq API key is missing")
        try:
            self._client = Groq(api_key=api_key)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Groq client: {e}") from e

        self._model = embedding_model

    async def embed(self, text: str) -> list[float]:
        try:
            response = await run_in_threadpool(
                self._client.embeddings.create,
                input=text,
                model=self._model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding via Groq: {e}")
            raise ProviderError(f"Groq - error generating embedding: {e}") from e
