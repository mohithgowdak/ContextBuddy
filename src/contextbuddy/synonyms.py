from __future__ import annotations

from typing import Dict, FrozenSet, List, Set

# Each group is a set of words that should be treated as equivalent for
# relevance scoring.  This is intentionally focused on the vocabulary that
# appears most often in business / technical / legal documents -- the primary
# use case for ContextBuddy.

_SYNONYM_GROUPS: List[FrozenSet[str]] = [
    # ── Vehicles / transport ──
    frozenset({"car", "automobile", "vehicle", "auto"}),
    frozenset({"truck", "lorry", "pickup"}),
    frozenset({"bus", "coach", "shuttle"}),
    frozenset({"flight", "airplane", "aircraft", "plane"}),
    frozenset({"ship", "vessel", "boat"}),

    # ── Money / payments ──
    frozenset({"payment", "remittance", "disbursement", "payout"}),
    frozenset({"cost", "price", "charge", "fee", "expense", "rate"}),
    frozenset({"salary", "wage", "compensation", "pay", "earnings", "income"}),
    frozenset({"invoice", "bill", "receipt", "statement"}),
    frozenset({"refund", "reimbursement", "rebate", "repayment"}),
    frozenset({"discount", "rebate", "markdown", "reduction"}),
    frozenset({"profit", "gain", "revenue", "earnings"}),
    frozenset({"loss", "deficit", "shortfall"}),
    frozenset({"budget", "allocation", "funding"}),
    frozenset({"tax", "levy", "duty", "tariff"}),
    frozenset({"debt", "liability", "obligation", "owing"}),
    frozenset({"loan", "credit", "mortgage", "advance"}),

    # ── Buy / sell ──
    frozenset({"buy", "purchase", "acquire", "procure", "order"}),
    frozenset({"sell", "vend", "trade", "retail"}),
    frozenset({"customer", "client", "buyer", "patron", "consumer"}),
    frozenset({"vendor", "supplier", "seller", "provider", "merchant"}),

    # ── Legal / contracts ──
    frozenset({"agreement", "contract", "deal", "pact", "arrangement"}),
    frozenset({"clause", "provision", "stipulation", "term", "condition"}),
    frozenset({"penalty", "fine", "sanction", "forfeiture"}),
    frozenset({"terminate", "cancel", "end", "revoke", "annul", "void"}),
    frozenset({"liable", "responsible", "accountable", "answerable"}),
    frozenset({"warranty", "guarantee", "assurance"}),
    frozenset({"comply", "conform", "adhere", "abide", "obey"}),
    frozenset({"dispute", "conflict", "disagreement", "contention"}),
    frozenset({"indemnify", "compensate", "reimburse"}),
    frozenset({"breach", "violation", "infringement"}),

    # ── Employment / HR ──
    frozenset({"employee", "worker", "staff", "personnel"}),
    frozenset({"employer", "company", "firm", "organization", "organisation"}),
    frozenset({"hire", "recruit", "employ", "engage", "onboard"}),
    frozenset({"fire", "terminate", "dismiss", "discharge", "layoff"}),
    frozenset({"resign", "quit", "leave", "depart"}),
    frozenset({"promotion", "advancement", "upgrade"}),
    frozenset({"benefit", "perk", "allowance", "entitlement"}),

    # ── Technology ──
    frozenset({"error", "bug", "defect", "fault", "issue", "glitch"}),
    frozenset({"fix", "patch", "repair", "resolve", "remedy"}),
    frozenset({"deploy", "release", "ship", "launch", "publish"}),
    frozenset({"server", "host", "machine", "instance", "node"}),
    frozenset({"database", "datastore", "repository", "store", "db"}),
    frozenset({"api", "endpoint", "interface", "service"}),
    frozenset({"authentication", "auth", "login", "signin"}),
    frozenset({"authorization", "permission", "access", "privilege"}),
    frozenset({"encrypt", "cipher", "encode", "protect"}),
    frozenset({"configuration", "config", "setting", "preference"}),
    frozenset({"latency", "delay", "lag", "response time"}),
    frozenset({"throughput", "bandwidth", "capacity"}),
    frozenset({"crash", "failure", "outage", "downtime"}),

    # ── General adjectives ──
    frozenset({"fast", "quick", "rapid", "swift", "speedy"}),
    frozenset({"slow", "sluggish", "gradual", "leisurely"}),
    frozenset({"cheap", "affordable", "inexpensive", "economical", "low-cost"}),
    frozenset({"expensive", "costly", "pricey", "premium"}),
    frozenset({"big", "large", "huge", "massive", "enormous"}),
    frozenset({"small", "tiny", "little", "miniature", "compact"}),
    frozenset({"important", "critical", "crucial", "vital", "essential", "key"}),
    frozenset({"easy", "simple", "straightforward", "effortless"}),
    frozenset({"hard", "difficult", "challenging", "tough", "complex"}),
    frozenset({"safe", "secure", "protected"}),
    frozenset({"dangerous", "risky", "hazardous", "unsafe"}),
    frozenset({"old", "outdated", "legacy", "deprecated", "obsolete"}),
    frozenset({"new", "modern", "latest", "current", "recent"}),

    # ── General verbs ──
    frozenset({"create", "make", "build", "generate", "produce", "construct"}),
    frozenset({"delete", "remove", "erase", "eliminate", "discard", "drop"}),
    frozenset({"update", "modify", "change", "alter", "revise", "amend", "edit"}),
    frozenset({"send", "transmit", "deliver", "dispatch", "forward"}),
    frozenset({"receive", "get", "obtain", "accept", "collect"}),
    frozenset({"start", "begin", "initiate", "commence", "launch"}),
    frozenset({"stop", "halt", "cease", "pause", "suspend"}),
    frozenset({"allow", "permit", "enable", "authorize", "grant"}),
    frozenset({"deny", "reject", "refuse", "decline", "block"}),
    frozenset({"verify", "validate", "confirm", "check", "authenticate"}),
    frozenset({"analyze", "examine", "inspect", "review", "assess", "evaluate"}),
    frozenset({"improve", "enhance", "optimize", "upgrade", "boost"}),
    frozenset({"reduce", "decrease", "lower", "minimize", "cut", "shrink"}),
    frozenset({"increase", "grow", "expand", "raise", "boost", "escalate"}),
    frozenset({"show", "display", "present", "exhibit", "demonstrate"}),
    frozenset({"hide", "conceal", "obscure", "mask"}),
    frozenset({"help", "assist", "support", "aid"}),
    frozenset({"fail", "malfunction", "break", "crash"}),
    frozenset({"store", "save", "persist", "retain", "keep", "archive"}),
    frozenset({"load", "fetch", "retrieve", "read", "pull"}),

    # ── General nouns ──
    frozenset({"problem", "issue", "bug", "defect", "flaw"}),
    frozenset({"solution", "fix", "resolution", "remedy", "answer"}),
    frozenset({"document", "file", "record", "paper", "report"}),
    frozenset({"meeting", "conference", "session", "call"}),
    frozenset({"deadline", "due date", "cutoff", "target date"}),
    frozenset({"team", "group", "squad", "department", "unit"}),
    frozenset({"manager", "supervisor", "lead", "director", "head"}),
    frozenset({"policy", "rule", "regulation", "guideline", "standard"}),
    frozenset({"goal", "objective", "target", "aim", "purpose"}),
    frozenset({"risk", "threat", "hazard", "danger", "vulnerability"}),
    frozenset({"result", "outcome", "output", "consequence", "effect"}),

    # ── Medical / health (common in support docs) ──
    frozenset({"doctor", "physician", "practitioner", "clinician"}),
    frozenset({"patient", "client", "individual", "subject"}),
    frozenset({"disease", "illness", "condition", "disorder", "ailment", "sickness"}),
    frozenset({"medicine", "medication", "drug", "treatment", "therapy", "remedy"}),
    frozenset({"symptom", "sign", "indication", "manifestation"}),
    frozenset({"diagnosis", "assessment", "evaluation", "finding"}),
]

# Build a fast lookup: word -> frozenset of all its synonyms
_SYNONYM_MAP: Dict[str, FrozenSet[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_MAP[_word] = _group


def expand_synonyms(word: str) -> FrozenSet[str]:
    """Return the synonym group for *word*, or a singleton set if unknown."""
    group = _SYNONYM_MAP.get(word.lower())
    if group is not None:
        return group
    return frozenset({word.lower()})


def expand_query_terms(terms: List[str]) -> Set[str]:
    """
    Given a list of stemmed query tokens, return an expanded set that
    includes all known synonyms for each token.
    """
    expanded: Set[str] = set()
    for t in terms:
        expanded.update(expand_synonyms(t))
    return expanded
