"""Wrapper around the Gemini LLM with retries, model fallback, caching and JSON support."""

import hashlib
import json
import logging
import re

from google import genai
from google.genai import errors, types
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    BASE_DIR,
    LLM_FALLBACK_MODELS,
    LLM_MODEL,
    TEMPERATURE,
    USE_LLM_CACHE,
    get_api_key,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_client: genai.Client | None = None
_CACHE_DIR = BASE_DIR / "data" / "llm_cache"


class GenerationError(Exception):
    """Raised when Gemini returns no usable text (blocked, empty, or cut off)."""


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            api_key=get_api_key(),
            http_options=types.HttpOptions(
                timeout=60_000,  # milliseconds; a stalled request fails instead of hanging
                retry_options=types.HttpRetryOptions(attempts=1),  # no hidden SDK retries
            ),
        )
    return _client


def _is_retryable(exc: BaseException) -> bool:
    """Retry rate limits (429) and temporary server errors, never bad requests."""
    return isinstance(exc, errors.APIError) and exc.code in (429, 500, 502, 503, 504)


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(2),   # give up on a busy model quickly; a backup exists
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_model(model: str, prompt: str, config: types.GenerateContentConfig):
    return _get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )


def _call_with_fallback(prompt: str, config: types.GenerateContentConfig):
    """Try the main model first, then each backup if it is overloaded or retired."""
    models = [LLM_MODEL] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL]
    last_exc: Exception | None = None

    for model in models:
        try:
            logger.info("Calling model %s", model)
            return _call_model(model, prompt, config)
        except errors.APIError as exc:
            if not (_is_retryable(exc) or exc.code == 404):
                raise                      # bad key / bad request: don't hide it
            logger.warning("Model %s failed with %s; trying next model", model, exc.code)
            last_exc = exc

    raise last_exc


def _extract_text(response) -> str:
    """Pull the text out of a response, or explain why there isn't any."""
    text = (response.text or "").strip()
    if text:
        return text

    reason = "unknown reason"
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        reason = f"prompt blocked ({response.prompt_feedback.block_reason})"
    elif response.candidates and response.candidates[0].finish_reason:
        reason = f"finish reason: {response.candidates[0].finish_reason}"
    raise GenerationError(f"Gemini returned no text ({reason}).")


def _generate(
    kind: str,
    prompt: str,
    system_instruction: str | None,
    temperature: float,
    config: types.GenerateContentConfig,
) -> str:
    """Return model text, reusing a saved answer for an identical request."""
    path = None
    if USE_LLM_CACHE:
        key = hashlib.sha256(
            json.dumps([kind, prompt, system_instruction, temperature]).encode()
        ).hexdigest()
        path = _CACHE_DIR / f"{key}.txt"
        if path.exists():
            logger.info("LLM cache hit")
            return path.read_text(encoding="utf-8")

    text = _extract_text(_call_with_fallback(prompt, config))

    if path is not None:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write LLM cache: %s", exc)
    return text


def generate_text(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = TEMPERATURE,
) -> str:
    """Send a prompt and return the model's text answer."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
    )
    return _generate("text", prompt, system_instruction, temperature, config)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if the model added them anyway."""
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def generate_json(
    prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.0,
):
    """Send a prompt and return the response parsed as JSON (a dict or list)."""
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime_type="application/json",
    )
    text = _generate("json", prompt, system_instruction, temperature, config)
    try:
        return json.loads(_strip_code_fences(text))
    except json.JSONDecodeError as exc:
        raise GenerationError(f"Model did not return valid JSON: {text[:200]}") from exc