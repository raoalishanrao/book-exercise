"""Chat LLM with Gemini primary and Groq fallback."""

import logging
import time

from google import genai
from google.genai.errors import APIError, ClientError, ServerError
from groq import Groq

from src.config import env_list, optional_env, require_env
from src.utils.google_errors import error_code, retry_delay_seconds

logger = logging.getLogger("tutor.llm")

GEMINI_CHAT_MODEL = "gemini-2.5-flash"
GEMINI_MAX_RETRIES = 3
RETRYABLE_GEMINI_CODES = {429, 500, 503, 504}


class ChatLLM:
    def __init__(self):
        self.gemini = genai.Client(api_key=require_env("GEMINI_API_KEY"))
        self.groq_key = optional_env("GROQ_API_KEY")
        self.groq_models = self._groq_model_chain()

    def generate(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        try:
            text = self._generate_gemini(system_prompt, user_prompt)
            logger.info("Gemini reply ok model=%s", GEMINI_CHAT_MODEL)
            return text, GEMINI_CHAT_MODEL
        except Exception as exc:
            code = error_code(exc)
            logger.warning(
                "Gemini failed code=%s error=%s",
                code,
                exc,
                exc_info=True,
            )
            if not self.groq_key:
                raise
            logger.info("Falling back to Groq models: %s", self.groq_models)
            return self._generate_groq(system_prompt, user_prompt)

    def _generate_gemini(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                response = self.gemini.models.generate_content(
                    model=GEMINI_CHAT_MODEL,
                    contents=prompt,
                )
                text = response.text
                if text:
                    return text
                raise RuntimeError("Empty Gemini response")
            except (ServerError, ClientError, APIError) as exc:
                code = error_code(exc)
                if code in RETRYABLE_GEMINI_CODES and attempt < GEMINI_MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.info(
                        "Gemini retry %s/%s in %ss (code=%s)",
                        attempt + 1,
                        GEMINI_MAX_RETRIES,
                        wait,
                        code,
                    )
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Gemini chat failed after retries")

    def _groq_model_chain(self) -> list[str]:
        models: list[str] = []
        primary = optional_env("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
        if primary:
            models.append(primary)
        for model in env_list("GROQ_CHAT_FALLBACK_MODELS"):
            if model not in models:
                models.append(model)
        return models

    def _generate_groq(self, system_prompt: str, user_prompt: str) -> tuple[str, str]:
        if not self.groq_models:
            raise RuntimeError("GROQ_API_KEY is set but no GROQ_CHAT_MODEL configured")

        client = Groq(api_key=self.groq_key)
        last_error: Exception | None = None

        for model in self.groq_models:
            try:
                logger.info("Groq request model=%s", model)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                    max_tokens=2048,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    logger.info("Groq reply ok model=%s", model)
                    return text, model
                last_error = RuntimeError(f"Empty Groq response from {model}")
                logger.warning("Groq empty response model=%s", model)
            except Exception as exc:
                logger.warning("Groq model %s failed: %s", model, exc, exc_info=True)
                last_error = exc

        raise RuntimeError(f"All Groq models failed: {last_error}")
