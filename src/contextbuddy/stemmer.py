from __future__ import annotations

import re
from typing import List

_VOWELS = frozenset("aeiou")
_word_re = re.compile(r"[a-z0-9]+")


def _has_vowel(stem: str) -> bool:
    return any(c in _VOWELS for c in stem)


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in _VOWELS


def _measure(stem: str) -> int:
    """Rough consonant-vowel sequence count (Porter's 'm')."""
    m = 0
    in_vowel = False
    for c in stem:
        is_v = c in _VOWELS
        if is_v and not in_vowel:
            in_vowel = True
        elif not is_v and in_vowel:
            m += 1
            in_vowel = False
    return m


def stem(word: str) -> str:
    """
    Lightweight suffix-stripping stemmer.

    Covers the highest-impact English morphology rules (plurals, verb forms,
    common derivational suffixes) in pure Python.  Not as thorough as a full
    Porter implementation, but good enough to make "payments" match "payment"
    and "running" match "run" without any dependencies.
    """
    w = word.lower().strip()
    if len(w) <= 2:
        return w

    # ── Step 1: plurals ──
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s") and not w.endswith("us") and not w.endswith("ss"):
        w = w[:-1]

    # ── Step 2: past tense / gerund ──
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    elif w.endswith("ed") and _has_vowel(w[:-2]):
        w = w[:-2]
        w = _step1b_cleanup(w)
    elif w.endswith("ing") and _has_vowel(w[:-3]):
        w = w[:-3]
        w = _step1b_cleanup(w)

    # ── Step 3: y -> i ──
    if w.endswith("y") and len(w) > 2 and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    # ── Step 4: derivational suffixes (long stems only) ──
    _step4_map = {
        "ational": "ate",
        "tional": "tion",
        "enci": "ence",
        "anci": "ance",
        "izer": "ize",
        "iser": "ise",
        "abli": "able",
        "alli": "al",
        "entli": "ent",
        "eli": "e",
        "ousli": "ous",
        "ization": "ize",
        "isation": "ise",
        "ation": "ate",
        "ator": "ate",
        "alism": "al",
        "iveness": "ive",
        "fulness": "ful",
        "ousness": "ous",
        "aliti": "al",
        "iviti": "ive",
        "biliti": "ble",
    }
    for suffix, replacement in _step4_map.items():
        if w.endswith(suffix):
            stem_part = w[: -len(suffix)]
            if _measure(stem_part) > 0:
                w = stem_part + replacement
            break

    # ── Step 5: common endings ──
    _step5_suffixes = [
        "ement", "ment", "ness", "able", "ible", "ful",
        "less", "ance", "ence", "ment", "ity", "ive",
        "ous", "ize", "ise", "ate", "al",
    ]
    for suffix in _step5_suffixes:
        if w.endswith(suffix):
            stem_part = w[: -len(suffix)]
            if _measure(stem_part) > 1:
                w = stem_part
            break

    # ── Step 6: trailing 'e' ──
    if w.endswith("e"):
        stem_part = w[:-1]
        if _measure(stem_part) > 1:
            w = stem_part
        elif _measure(stem_part) == 1 and not (
            len(stem_part) >= 3
            and stem_part[-1] not in _VOWELS
            and stem_part[-2] in _VOWELS
            and stem_part[-3] not in _VOWELS
        ):
            w = stem_part

    # ── Step 7: double consonant cleanup ──
    if _ends_double_consonant(w) and w[-1] not in "lsz" and _measure(w[:-1]) > 1:
        w = w[:-1]

    return w


def _step1b_cleanup(w: str) -> str:
    if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
        w += "e"
    elif _ends_double_consonant(w) and w[-1] not in "lsz":
        w = w[:-1]
    elif _measure(w) == 1 and len(w) >= 3 and w[-1] not in _VOWELS and w[-2] in _VOWELS and w[-3] not in _VOWELS:
        w += "e"
    return w


def tokenize_and_stem(text: str) -> List[str]:
    """Lowercase, split into word tokens, and stem each one."""
    return [stem(m.group(0)) for m in _word_re.finditer(text.lower())]
