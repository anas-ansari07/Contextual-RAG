import os
import requests


CONTEXT_PROMPT = """
You are preparing a document for a retrieval-augmented generation
(RAG) knowledge base.

DOCUMENT CONTEXT:
{document_context}

SELECTED CHUNK:
{chunk_text}

Write a concise contextual description of the selected chunk.

Include, when supported by the source:
- The topic or purpose of this passage
- The relevant system, entity, process, or concept
- The relationship to its surrounding section
- Important terms that would help retrieve this passage

Use only information supported by the provided document.
Do not answer a question.
Do not invent facts.
Keep the description to 2-4 sentences.

Return only the contextual description.
"""



def generate_context(chunk_text: str, document_context: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    prompt = CONTEXT_PROMPT.format(
        document_context=document_context,
        chunk_text=chunk_text,
    )

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Title": "ContextIQ",
        },
        
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 500,
            "reasoning": {
                "enabled": False
            },
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        },

        timeout=90,
    )

    # Include OpenRouter's error response when the request fails.
    if not response.ok:
        raise RuntimeError(
            f"OpenRouter returned HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()
    choices = data.get("choices") or []

    if not choices:
        raise RuntimeError(
            f"OpenRouter returned no choices. Response: "
            f"{str(data)[:800]}"
        )

    choice = choices[0]
    message = choice.get("message") or {}
    result = message.get("content")

    if isinstance(result, list):
        result = " ".join(
            item.get("text", "")
            for item in result
            if isinstance(item, dict)
        )

    if not isinstance(result, str) or not result.strip():
        finish_reason = choice.get("finish_reason")
        returned_model = data.get("model")
        usage = data.get("usage")

        raise RuntimeError(
            "OpenRouter returned empty context. "
            f"Model={returned_model}, "
            f"finish_reason={finish_reason}, "
            f"usage={usage}, "
            f"message={str(message)[:500]}"
        )

    return result.strip()
