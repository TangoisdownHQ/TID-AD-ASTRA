import os

import requests


OPENAI_API_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/responses")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")


def openai_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def render_openai_answer(user_message: str, deterministic_payload: dict, session_context: dict | None = None):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    system_prompt = (
        "You are the TID-AD-ASTRA mission chat layer. "
        "Answer strictly from the structured planet-catalog data provided to you. "
        "Do not invent planets, measurements, or sources. "
        "If data is missing, say it is missing. "
        "Keep answers concise and useful for a CLI."
    )

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"User message: {user_message}\n\n"
                            f"Session context: {session_context or {}}\n\n"
                            f"Structured result: {deterministic_payload}\n\n"
                            "Rewrite this into a direct CLI answer. "
                            "If the structured answer already looks good, improve phrasing only."
                        ),
                    }
                ],
            },
        ],
    }

    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()

    output_text = data.get("output_text")
    if output_text:
        return output_text.strip()

    outputs = data.get("output") or []
    texts = []
    for item in outputs:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()
