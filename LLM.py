"""
LLM service: wraps Google Gemini 1.5 Flash API.
Builds grounded prompts from retrieved context + conversation history.
"""

import logging
from typing import Optional
import google.generativeai as genai
from config import settings

logger = logging.getLogger(__name__)

# Configure Gemini once
if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)

# ─── Prompt Template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional banking support assistant for a fintech company.

Your responsibilities:
- Answer customer questions about loans, credit cards, banking policies, and financial services
- Provide accurate, grounded information based ONLY on the context provided below

STRICT RULES:
1. Answer ONLY based on the provided CONTEXT. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say exactly:
   "I could not find enough information in the uploaded documents to answer this. Please contact our support team for assistance."
3. NEVER fabricate interest rates, fees, limits, or policy details.
4. Be concise, professional, and customer-friendly.
5. If the user refers to something from earlier in the conversation (e.g. "what about it?", "what is the rate for it?"), 
   use the CONVERSATION HISTORY to understand what they are referring to.
6. Format your response clearly. Use bullet points for lists of features or conditions.

CONTEXT FROM KNOWLEDGE BASE:
{context}

CONVERSATION HISTORY:
{history}
"""


def _format_history(history: list[dict]) -> str:
    """Format session history into a readable string for the prompt."""
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "Customer" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


async def generate_response(
    user_message: str,
    context: str,
    history: list[dict],
) -> str:
    """
    Generate a grounded LLM response using Gemini 1.5 Flash.

    Args:
        user_message: The current user query
        context: Retrieved document chunks (formatted string)
        history: Conversation history list of {"role": ..., "content": ...}

    Returns:
        Assistant's response string
    """
    if not settings.gemini_api_key:
        return _fallback_no_api_key()

    system_text = SYSTEM_PROMPT.format(
        context=context or "No context retrieved.",
        history=_format_history(history),
    )

    full_prompt = f"{system_text}\n\nCustomer: {user_message}\nAssistant:"

    try:
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.2,      # Low temp = factual, less creative
                "top_p": 0.8,
                "max_output_tokens": 800,
            },
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ],
        )

        response = model.generate_content(full_prompt)
        text = response.text.strip()
        return text if text else _fallback_empty_response()

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        # Retry once
        try:
            response = model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e2:
            logger.error(f"Gemini retry failed: {e2}")
            return _fallback_api_error()


def _fallback_no_api_key() -> str:
    return (
        "⚠️ The AI service is not configured. "
        "Please set the GEMINI_API_KEY environment variable."
    )


def _fallback_empty_response() -> str:
    return (
        "I was unable to generate a response. "
        "Please rephrase your question and try again."
    )


def _fallback_api_error() -> str:
    return (
        "The AI service is temporarily unavailable. "
        "Please try again in a moment or contact our support team."
    )
