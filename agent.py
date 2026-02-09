import os
import json
import copy
import re
import time
import requests
from typing import Optional
from dotenv import load_dotenv
from logger import logger


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class Agent:
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_name: str, system_prompt: str = ""):
        self.system_prompt: str = system_prompt
        self.model_name: str = model_name
        self.conversation_history: list[str] = []
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        }

    def reset_conversation(self):
        """Clear conversation history to start fresh."""
        self.conversation_history = []

    def call_agent(self, user_prompt: str, preserve_conversation_history: bool = False):
        if self.system_prompt and len(self.conversation_history) == 0:
            system_message = {
                "role": "system",
                "content": self.system_prompt
            }
            self.conversation_history.append(system_message)

        new_message = {
            "role": "user",
            "content": user_prompt
        }
        messages = copy.deepcopy(self.conversation_history)
        messages.append(new_message)

        data = json.dumps({
            "model": self.model_name,
            "messages": messages
        })

        try:
            response = requests.post(
                url=self.OPENROUTER_URL,
                headers=self.headers,
                data=data
            )

            res = response.json()
            if "choices" not in res.keys():
                raise KeyError("Choices not present in response")
            model_response = res["choices"][0]["message"]
            if preserve_conversation_history:
                self.conversation_history.append(new_message)
                self.conversation_history.append(model_response)

            text_response = model_response["content"]
            return text_response
        except Exception as e:
            logger.info(f"Exception occurred: {e}")
            return ''

    def call_agent_with_retry(
        self,
        user_prompt: str,
        preserve_conversation_history: bool = False,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> str:
        """
        Call the agent with exponential backoff retry logic.

        Args:
            user_prompt: The prompt to send
            preserve_conversation_history: Whether to preserve history
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff

        Returns:
            The agent's response, or empty string on failure
        """
        for attempt in range(max_retries):
            result = self.call_agent(user_prompt, preserve_conversation_history)
            if result:
                return result

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retry {attempt + 1}/{max_retries} after {delay}s delay")
                time.sleep(delay)

        return ''

    @staticmethod
    def parse_json_response(response: str) -> Optional[dict | list]:
        """
        Extract and parse JSON from an LLM response.
        Handles responses with markdown code blocks or raw JSON.

        Args:
            response: The LLM response string

        Returns:
            Parsed JSON as dict/list, or None if parsing fails
        """
        if not response:
            return None

        # Try to find JSON in code blocks first
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, response)

        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try parsing the raw response
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON-like content (starts with { or [)
        json_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_pattern, response)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None