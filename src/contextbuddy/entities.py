from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Set, Sequence


# ── Built-in patterns ──────────────────────────────────────────────
_PATTERNS: Dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "url": re.compile(r"\bhttps?://[^\s)\"'>]+", re.IGNORECASE),
    "iso_date": re.compile(r"\b\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?\b"),
    "us_date": re.compile(r"\b(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/(?:19|20)\d{2}\b"),
    "uuid": re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
        re.IGNORECASE,
    ),
    "ticket": re.compile(r"\b[A-Z]{2,10}-\d{1,7}\b"),
    "phone": re.compile(
        r"(?:\+\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b"
    ),
    "money": re.compile(
        r"(?:\$|€|£|¥|₹)\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s?(?:USD|EUR|GBP|JPY|INR)\b",
        re.IGNORECASE,
    ),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "version": re.compile(r"\bv?\d+\.\d+(?:\.\d+)+\b"),
    # Domain-ish identifiers (common in real tickets/contracts)
    "policy": re.compile(r"\b[A-Z]{2,}-\d{4,}-[A-Z0-9]+\b"),
    # git SHAs (40-hex) show up in incident reports and PR discussions
    "git_sha": re.compile(r"\b[0-9a-f]{40}\b", re.IGNORECASE),
}

_id_re = re.compile(
    r"\b(?:id|uid|user_id|account_id|order_id|invoice_id|customer_id|"
    r"session_id|transaction_id|ref|reference|sku|arn)"
    r"\s*[:=#]?\s*([A-Za-z0-9_-]{4,})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EntityExtractor:
    """
    Extracts structured entities from text.

    Recognised categories (all regex-based, zero dependencies):
    email, url, iso_date, us_date, uuid, ticket, phone, money, ipv4, version,
    plus a catch-all for id-like key=value patterns.
    """

    max_entities: int = 50
    enabled_patterns: Sequence[str] = tuple(_PATTERNS.keys())

    def extract(self, text: str) -> List[str]:
        found: List[str] = []
        seen: Set[str] = set()

        def _add(value: str) -> bool:
            v = value.strip()
            if v and v not in seen:
                seen.add(v)
                found.append(v)
            return len(found) >= self.max_entities

        for name in self.enabled_patterns:
            pat = _PATTERNS.get(name)
            if not pat:
                continue
            for m in pat.finditer(text):
                if _add(m.group(0)):
                    return found

        for m in _id_re.finditer(text):
            val = m.group(1)
            if _add(val):
                break

        return found
