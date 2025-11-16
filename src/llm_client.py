"""
LLM Client - Gọi LLM API (hosted)
"""
import requests
import time
from typing import List, Dict, Optional
from src.config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MAX_RETRIES,
    RETRY_DELAY,
    AGENT_TIMEOUT,
    DEBUG
)


def call_llm(
    messages: List[Dict],
    temperature: float = None,
    max_tokens: int = None,
    timeout: int = None
) -> str:
    """
    Gọi LLM API để nhận response
    
    Args:
        messages: List of message dicts với format:
            [{"role": "system"|"user"|"assistant", "content": "..."}]
        temperature: Temperature cho generation (mặc định từ config)
        max_tokens: Max tokens cho response (mặc định từ config)
        timeout: Request timeout in seconds (mặc định từ config)
    
    Returns:
        str: Response content từ LLM
    
    Raises:
        ValueError: Nếu config không hợp lệ
        requests.RequestException: Nếu API call thất bại
    """
    # Validate config
    if not LLM_API_URL:
        raise ValueError("LLM_API_URL is not configured")
    if not LLM_MODEL_NAME:
        raise ValueError("LLM_MODEL_NAME is not configured")
    
    # Use defaults from config if not provided
    if temperature is None:
        temperature = LLM_TEMPERATURE
    if max_tokens is None:
        max_tokens = LLM_MAX_TOKENS
    if timeout is None:
        timeout = AGENT_TIMEOUT
    
    # Prepare payload
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
    }
    
    # Add authorization if API key is provided
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    
    # Retry logic
    last_exception = None
    for attempt in range(MAX_RETRIES):
        try:
            if DEBUG and attempt > 0:
                print(f"Retrying LLM API call (attempt {attempt + 1}/{MAX_RETRIES})...")
            
            # Make API request
            resp = requests.post(
                LLM_API_URL,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response content (hỗ trợ nhiều format)
            content = _extract_content(data)
            
            if DEBUG:
                print(f"LLM API call successful (tokens: ~{len(content.split())})")
            
            return content
            
        except requests.exceptions.Timeout:
            last_exception = f"Request timeout after {timeout}s"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
            continue
            
        except requests.exceptions.HTTPError as e:
            last_exception = f"HTTP error {e.response.status_code}: {e.response.text}"
            # Don't retry on client errors (4xx)
            if 400 <= e.response.status_code < 500:
                raise requests.RequestException(last_exception) from e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue
            
        except requests.exceptions.RequestException as e:
            last_exception = f"Request error: {str(e)}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue
            
        except (KeyError, IndexError) as e:
            last_exception = f"Unexpected response format: {str(e)}"
            raise ValueError(f"Failed to parse LLM response: {last_exception}") from e
    
    # All retries failed
    raise requests.RequestException(
        f"LLM API call failed after {MAX_RETRIES} attempts. Last error: {last_exception}"
    )


def _extract_content(data: dict) -> str:
    """
    Extract content từ response data (hỗ trợ nhiều format API)
    
    Args:
        data: JSON response từ API
    
    Returns:
        str: Extracted content
    """
    # OpenAI format: data["choices"][0]["message"]["content"]
    if "choices" in data and len(data["choices"]) > 0:
        choice = data["choices"][0]
        if "message" in choice:
            return choice["message"].get("content", "")
        elif "text" in choice:
            return choice["text"]
    
    # Anthropic format: data["content"][0]["text"]
    if "content" in data and isinstance(data["content"], list):
        if len(data["content"]) > 0 and "text" in data["content"][0]:
            return data["content"][0]["text"]
    
    # Ollama format: data["response"]
    if "response" in data:
        return data["response"]
    
    # Generic format: data["message"] or data["text"]
    if "message" in data:
        return data["message"]
    if "text" in data:
        return data["text"]
    
    # Fallback: return first string value found
    for key, value in data.items():
        if isinstance(value, str) and value:
            return value
    
    raise ValueError(f"Could not extract content from response: {data}")


def call_llm_simple(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Wrapper đơn giản để gọi LLM với một prompt
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
    
    Returns:
        str: Response từ LLM
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    return call_llm(messages)

