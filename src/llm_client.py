"""
LLM Client Module for ZHIPU AI (GLM-4-Flash).

This module provides a client for interacting with the ZHIPU AI API
to generate answers based on retrieved context.
"""

from typing import Dict, Any, Optional
from zhipuai import ZhipuAI

from src.utils import get_logger

logger = get_logger(__name__)


class GLMClient:
    """
    Client for ZHIPU AI GLM-4-Flash API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.5-flash",
        temperature: float = 0.7,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the GLM client.

        Args:
            api_key: ZHIPU AI API key
            model: Model name (default: glm-4-flash)
            temperature: Default sampling temperature (can be overridden in generate)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.model = model
        self.default_temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

        logger.info(f"Initializing GLMClient with model={model}, temperature={temperature}")

        try:
            self.client = ZhipuAI(api_key=api_key)
            logger.info("GLMClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GLMClient: {e}")
            raise

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate answer using GLM-4-Flash.

        Args:
            query: User query
            context: Retrieved context from chunks
            system_prompt: System prompt for the LLM
            temperature: Sampling temperature (uses default if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated answer or None if failed
        """
        # Use default temperature if not specified
        if temperature is None:
            temperature = self.default_temperature

        logger.info("Generating answer with GLM-4-Flash")
        logger.debug(f"Query: {query[:100]}...")
        logger.debug(f"Context length: {len(context)} characters")

        # Construct messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        try:
            # Call API
            logger.debug(f"Calling ZHIPU API with model={self.model}, temperature={temperature}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Extract answer
            answer = response.choices[0].message.content
            logger.info("Successfully generated answer")
            logger.debug(f"Answer length: {len(answer)} characters")

            return answer

        except Exception as e:
            logger.error(f"Failed to generate answer: {e}", exc_info=True)
            return None

    def generate_simple(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate response for a simple prompt (without RAG context).

        Args:
            prompt: User prompt
            temperature: Sampling temperature (uses default if None)
            max_tokens: Maximum tokens in response

        Returns:
            Generated response or None if failed
        """
        # Use default temperature if not specified
        if temperature is None:
            temperature = self.default_temperature

        logger.info("Generating simple response")
        logger.debug(f"Prompt: {prompt[:100]}...")

        messages = [
            {"role": "user", "content": prompt}
        ]

        try:
            logger.debug(f"Calling ZHIPU API with model={self.model}, temperature={temperature}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content
            logger.info("Successfully generated response")

            return answer

        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            return None
