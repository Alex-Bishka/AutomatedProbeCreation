import os
import json
import copy
import requests
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