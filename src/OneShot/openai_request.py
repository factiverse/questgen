"""Module containing the OpenAI Summary Generator class."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv

# from openai import AzureOpenAI


class OpenAIUtils:
    def __init__(self) -> None:
        """Initializes the OpenAI API and Elasticsearch client.

        Args:
            index: Elasticsearch index name
        """
        load_dotenv()
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
        openai.api_base = os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )  # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"  # this might change in the future
        self._engine = os.getenv("AZURE_OPENAI_MODEL_ID")

    def generate(self, prompt: str, model=None) -> str:
        """Generates a response from the OpenAI API for the given prompt.

        Args:
            prompt: The prompt to send to the API.
            model: The model to use for generating the response. Default is None.

        Returns:
            The API's response as a string.
        """
        if model is None:
            model = self._engine
        response = openai.ChatCompletion.create(
            engine=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
