from __future__ import annotations

from .types import ModelPricing

# ── OpenAI ──────────────────────────────────────────────────────────
OPENAI_GPT4O = ModelPricing(input_per_1k=0.0025, output_per_1k=0.010)
OPENAI_GPT4O_MINI = ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006)
OPENAI_GPT41 = ModelPricing(input_per_1k=0.002, output_per_1k=0.008)
OPENAI_GPT41_MINI = ModelPricing(input_per_1k=0.0004, output_per_1k=0.0016)
OPENAI_GPT41_NANO = ModelPricing(input_per_1k=0.0001, output_per_1k=0.0004)
OPENAI_O3 = ModelPricing(input_per_1k=0.002, output_per_1k=0.008)
OPENAI_O3_MINI = ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006)
OPENAI_O4_MINI = ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006)

# ── Anthropic ───────────────────────────────────────────────────────
CLAUDE_OPUS_4 = ModelPricing(input_per_1k=0.015, output_per_1k=0.075)
CLAUDE_SONNET_4 = ModelPricing(input_per_1k=0.003, output_per_1k=0.015)
CLAUDE_HAIKU_35 = ModelPricing(input_per_1k=0.0008, output_per_1k=0.004)

# ── Google ──────────────────────────────────────────────────────────
GEMINI_25_PRO = ModelPricing(input_per_1k=0.00125, output_per_1k=0.010)
GEMINI_25_FLASH = ModelPricing(input_per_1k=0.00015, output_per_1k=0.0006)
GEMINI_20_FLASH = ModelPricing(input_per_1k=0.0001, output_per_1k=0.0004)

# ── Meta (local) ───────────────────────────────────────────────────
LOCAL_FREE = ModelPricing(input_per_1k=0.0, output_per_1k=0.0)

PRESETS = {
    "gpt-4o": OPENAI_GPT4O,
    "gpt-4o-mini": OPENAI_GPT4O_MINI,
    "gpt-4.1": OPENAI_GPT41,
    "gpt-4.1-mini": OPENAI_GPT41_MINI,
    "gpt-4.1-nano": OPENAI_GPT41_NANO,
    "o3": OPENAI_O3,
    "o3-mini": OPENAI_O3_MINI,
    "o4-mini": OPENAI_O4_MINI,
    "claude-opus-4": CLAUDE_OPUS_4,
    "claude-sonnet-4": CLAUDE_SONNET_4,
    "claude-haiku-3.5": CLAUDE_HAIKU_35,
    "gemini-2.5-pro": GEMINI_25_PRO,
    "gemini-2.5-flash": GEMINI_25_FLASH,
    "gemini-2.0-flash": GEMINI_20_FLASH,
    "local": LOCAL_FREE,
}


def get_pricing(model_name: str) -> ModelPricing:
    """Look up pricing by model name. Falls back to gpt-4o-mini."""
    key = model_name.lower().strip()
    if key in PRESETS:
        return PRESETS[key]
    for name, pricing in PRESETS.items():
        if name in key or key in name:
            return pricing
    return OPENAI_GPT4O_MINI
