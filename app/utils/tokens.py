"""Tokenizer-aware token counting with a deterministic fallback.

Prefers a tokenizer compatible with the active model family when one is
installed (``tiktoken``, if present, gives a much closer approximation for
GPT/Llama-style BPE tokenizers than a character count). When no tokenizer
package is available -- the default in this environment -- a deterministic
fallback estimator is used instead so behavior never depends on network
access or model downloads.

The fallback is intentionally conservative: it takes the *larger* of a
character-based estimate (~4 chars/token, a widely used approximation for
English text) and a whitespace-token count, so it never meaningfully
undercounts short, punctuation-heavy inputs (e.g. secrets, JSON, code).
"""

from __future__ import annotations

import re

try:  # pragma: no cover - exercised only when tiktoken is installed
    import tiktoken
except ImportError:  # pragma: no cover - default state in this environment
    tiktoken = None

_ENCODING_NAME = 'cl100k_base'
_FALLBACK_CHARS_PER_TOKEN = 4
_WORD_PATTERN = re.compile(r'\S+')


def _load_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding(_ENCODING_NAME)
    except Exception:  # pragma: no cover - defensive, e.g. offline encoding fetch
        return None


_encoder = _load_encoder()


def tokenizer_backend() -> str:
    """Return which counting strategy is active, for observability/logging."""

    return 'tiktoken:' + _ENCODING_NAME if _encoder is not None else 'fallback:chars-per-4'


def estimate_tokens(text: str | None) -> int:
    """Estimate the token count of ``text`` using the best available strategy."""

    if not text:
        return 0
    if _encoder is not None:
        try:
            return len(_encoder.encode(text))
        except Exception:  # pragma: no cover - defensive fallback on encode errors
            pass
    return _fallback_estimate(text)


def _fallback_estimate(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    char_based = -(-len(stripped) // _FALLBACK_CHARS_PER_TOKEN)  # ceil division
    word_based = len(_WORD_PATTERN.findall(stripped))
    return max(char_based, word_based)
