from __future__ import annotations

from typing import List


_STRIP_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"}


def load_url(url: str, *, timeout: float = 15.0) -> List[str]:
    """
    Fetch a URL and extract clean text content.

    Requires: pip install "contextbuddy[web]"
    """
    try:
        import httpx  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Web loading requires 'httpx'. "
            "Install with: pip install \"contextbuddy[web]\""
        ) from e

    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Web loading requires 'beautifulsoup4'. "
            "Install with: pip install \"contextbuddy[web]\""
        ) from e

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup.find_all(list(_STRIP_TAGS)):
        tag.decompose()

    # Prefer <article> or <main> content; fall back to <body>.
    main = soup.find("article") or soup.find("main") or soup.find("body") or soup
    text = main.get_text(separator="\n", strip=True)

    import re
    paragraphs = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in paragraphs if len(p.strip()) > 30]
