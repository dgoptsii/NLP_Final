#!/usr/bin/env python3
"""
parse_mailbox.py

Reusable mailbox parser for both:
- Nazario phishing mbox
- Enron mbox created from create_mailbox.py

----------
1. Read any .mbox file
2. Extract subject + body safely
3. Handle MIME multipart emails
4. Preserve useful phishing evidence from HTML
5. Compute structural features
6. Remove near-duplicates more carefully
7. Save the result to CSV
8. Print detailed parsing statistics
"""

from __future__ import annotations

import argparse
import csv
import html
import mailbox
import re
from collections import Counter
from email.header import decode_header, make_header
from pathlib import Path
from statistics import mean
from urllib.parse import urlparse


# ============================================================================
# REGEXES
# ============================================================================

# Remove inline email-style headers that sometimes appear inside forwarded or
# quoted message bodies. Example:
# "From: John Doe"
# "Subject: Meeting"
INLINE_HEADER_RE = re.compile(
    r"^(from|to|cc|bcc|subject|date|sent|reply-to|message-id):\s+.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Remove HTML comments: <!-- ... -->
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Remove entire <script>...</script> and <style>...</style> blocks.
SCRIPT_STYLE_RE = re.compile(
    r"<(script|style).*?>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# Generic HTML tag remover: <...>
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Convert <br> and variants into line breaks.
BR_RE = re.compile(r"<\s*br\s*/?>", re.IGNORECASE)

# Convert closing </p> into line breaks.
P_RE = re.compile(r"</\s*p\s*>", re.IGNORECASE)

# Remove list tags such as <ul>, <ol>, <li>.
LIST_TAG_RE = re.compile(r"</?\s*(ul|ol|li)\b[^>]*>", re.IGNORECASE)

# Detect image tags in HTML.
IMG_RE = re.compile(r"<\s*img\b", re.IGNORECASE)

# Capture useful link-bearing attributes directly from HTML.
# This helps preserve hidden phishing links even if anchor text looks harmless.
HREF_RE = re.compile(r'href\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
SRC_RE = re.compile(r'src\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
ACTION_RE = re.compile(r'action\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

# Remove quoted reply lines that start with ">".
QUOTE_RE = re.compile(r"^\s*>.*$", re.MULTILINE)

# Detect common forwarded-message separators.
FORWARDED_RE = re.compile(
    r"(^[- ]*original message[- ]*$|^begin forwarded message:|^forwarded by )",
    re.IGNORECASE | re.MULTILINE,
)

# Remove email signatures beginning with the standard "--" separator.
SIGNATURE_RE = re.compile(r"\n(--\s*\n.*)$", re.DOTALL)

# Collapse repeated whitespace into one space.
MULTISPACE_RE = re.compile(r"\s+")

# Replace HTML nonbreaking spaces.
HTML_ENTITY_SPACE_RE = re.compile(r"&nbsp;?", re.IGNORECASE)

# Match visible URLs in text.
URL_RE = re.compile(
    r"(https?://[^\s<>'\"()]+|www\.[^\s<>'\"()]+)",
    re.IGNORECASE,
)

# Match email addresses.
EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Match IPv4 addresses.
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# Match numbers and numeric patterns.
NUM_RE = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")

# Detect URLs where the hostname itself is an IP address.
IP_URL_RE = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/|$)",
    re.IGNORECASE,
)

# Common phishing-related keywords.
PHISH_KEYWORDS = {
    "account", "verify", "verification", "suspend", "suspended", "confirm",
    "security", "password", "bank", "login", "update", "urgent", "click",
    "limited", "access", "identity", "billing", "alert", "unlock",
    "signin", "ssn", "validate", "dear", "customer", "immediately",
    "restricted", "warning", "notify", "failure", "expired",
}


# ============================================================================
# SMALL HELPERS
# ============================================================================

def header_to_str(value) -> str:
    """
    Safely convert an email header or payload-like object to a string.
    - email headers can be encoded in MIME form
    - some parsed objects are not plain strings
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        try:
            return str(value)
        except Exception:
            return ""


def normalize_url(url: str) -> str:
    """
    Normalize a URL so later parsing is more consistent.
    If it starts with 'www.', prepend 'http://'.
    """
    url = header_to_str(url).strip()
    if not url:
        return ""
    if url.lower().startswith("www."):
        return "http://" + url
    return url


def safe_netloc(url: str) -> str:
    """
    Extract the domain / netloc from a URL safely.
    Returns lowercase hostname-like string, or empty string on failure.
    """
    try:
        netloc = urlparse(url).netloc.lower().strip()
        return netloc
    except Exception:
        return ""


# ============================================================================
# HTML PROCESSING
# ============================================================================

def extract_links_from_html(raw_html: str) -> list[str]:
    """
    Extract href/src/action attribute values from raw HTML.
    - <a href="...">
    - <img src="...">
    - <form action="...">
    Plain text extraction alone can miss those.
    """
    links = []
    for regex in (HREF_RE, SRC_RE, ACTION_RE):
        links.extend(regex.findall(raw_html))

    out = []
    for link in links:
        link = normalize_url(link)
        if link:
            out.append(link)
    return out


def html_to_text_and_links(raw_html: str) -> tuple[str, list[str]]:
    """
    Convert HTML into readable text and separately return links extracted from
    link-bearing attributes.

    Steps:
    1. collect URLs both from visible text and HTML attributes
    2. remove comments/scripts/styles
    3. convert some tags into spaces/newlines
    4. strip all remaining tags
    5. unescape HTML entities
    """
    if not raw_html:
        return "", []

    visible_urls = [normalize_url(u) for u in URL_RE.findall(raw_html)]
    attr_links = extract_links_from_html(raw_html)
    all_links = [u for u in (visible_urls + attr_links) if u]

    text = COMMENT_RE.sub(" ", raw_html)
    text = SCRIPT_STYLE_RE.sub(" ", text)
    text = BR_RE.sub("\n", text)
    text = P_RE.sub("\n", text)
    text = LIST_TAG_RE.sub(" ", text)
    text = HTML_ENTITY_SPACE_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = MULTISPACE_RE.sub(" ", text).strip()

    return text, all_links


def decode_part(part) -> str:
    """
    Decode one MIME part into a string.

    Uses declared charset when available.
    Falls back to UTF-8 with replacement on errors.
    """
    payload = part.get_payload(decode=True)

    if payload is None:
        raw = part.get_payload()
        return header_to_str(raw)

    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        return payload.decode("utf-8", errors="replace")


# ============================================================================
# MESSAGE EXTRACTION
# ============================================================================

def extract_subject_body_meta(msg) -> tuple[str, str, dict]:
    """
    Extract:
    - subject
    - combined body text
    - metadata flags
    - extracted links

    If both plain text and HTML are present, we combine them instead of discarding HTML entirely.
    """
    subject = header_to_str(msg.get("subject", ""))

    plain_parts = []
    html_text_parts = []
    all_links = []

    has_html = 0
    has_form = 0
    has_script = 0
    has_iframe = 0
    has_embedded_images = 0
    num_attachments = 0

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()

            # Count explicit attachments and skip their content.
            if "attachment" in disp:
                num_attachments += 1
                continue

            # Detect inline/embedded images.
            if ctype.startswith("image/"):
                has_embedded_images = 1
                continue

            # Skip application payloads such as PDFs or binaries.
            if ctype.startswith("application/"):
                continue

            if ctype == "text/plain":
                txt = decode_part(part)
                if txt:
                    plain_parts.append(txt)

            elif ctype == "text/html":
                raw_html = decode_part(part)
                if raw_html:
                    has_html = 1
                    html_text, html_links = html_to_text_and_links(raw_html)
                    if html_text:
                        html_text_parts.append(html_text)
                    all_links.extend(html_links)

                    if re.search(r"<\s*form\b", raw_html, re.IGNORECASE):
                        has_form = 1
                    if re.search(r"<\s*script\b", raw_html, re.IGNORECASE):
                        has_script = 1
                    if re.search(r"<\s*iframe\b", raw_html, re.IGNORECASE):
                        has_iframe = 1
                    if IMG_RE.search(raw_html):
                        has_embedded_images = 1
    else:
        ctype = msg.get_content_type()
        payload = decode_part(msg)

        if ctype == "text/plain":
            if payload:
                plain_parts.append(payload)

        elif ctype == "text/html":
            if payload:
                has_html = 1
                html_text, html_links = html_to_text_and_links(payload)
                if html_text:
                    html_text_parts.append(html_text)
                all_links.extend(html_links)

                if re.search(r"<\s*form\b", payload, re.IGNORECASE):
                    has_form = 1
                if re.search(r"<\s*script\b", payload, re.IGNORECASE):
                    has_script = 1
                if re.search(r"<\s*iframe\b", payload, re.IGNORECASE):
                    has_iframe = 1
                if IMG_RE.search(payload):
                    has_embedded_images = 1

    body_plain = "\n".join(header_to_str(x) for x in plain_parts if x).strip()
    body_html = "\n".join(header_to_str(x) for x in html_text_parts if x).strip()

    # Combine plain and HTML text if both exist and are not identical.
    if body_plain and body_html:
        if body_plain.strip() == body_html.strip():
            body = body_plain
        else:
            body = f"{body_plain}\n\n{body_html}"
    else:
        body = body_plain if body_plain else body_html

    meta = {
        "has_html": has_html,
        "has_form_tag": has_form,
        "has_script_tag": has_script,
        "has_iframe_tag": has_iframe,
        "has_embedded_images": has_embedded_images,
        "num_attachments": num_attachments,
        "html_extracted_links": all_links,
    }

    return subject.strip(), body.strip(), meta


# ============================================================================
# TEXT CLEANING
# ============================================================================

def strip_replies_signature(text: str) -> str:
    """
    Remove common reply/forward/signature sections.
    """
    text = FORWARDED_RE.split(text)[0]
    text = SIGNATURE_RE.sub("", text)
    text = QUOTE_RE.sub("", text)
    return text


def clean_text(text: str) -> str:
    """
    Basic text cleanup:
    - normalize line breaks
    - unescape HTML entities
    - remove inline forwarded-message headers
    - strip signatures/replies
    - collapse whitespace
    """
    text = header_to_str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html.unescape(text)
    text = INLINE_HEADER_RE.sub(" ", text)
    text = strip_replies_signature(text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def mask_text(text: str) -> str:
    """
    Replace specific surface forms with placeholders.

    - reduce overfitting to exact URLs, emails, account numbers, IDs
    - keep the general pattern of the message
    """
    text = clean_text(text)
    text = URL_RE.sub(" <URL> ", text)
    text = EMAIL_RE.sub(" <EMAIL> ", text)
    text = IP_RE.sub(" <IP> ", text)
    text = NUM_RE.sub(" <NUM> ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip().lower()


def debias_text(text: str) -> str:
    """
    More aggressive normalization on top of mask_text().

    Removes:
    - long hex-like strings
    - very long tokens
    - some corpus-specific tokens
    """
    text = mask_text(text)
    text = re.sub(r"\b[a-f0-9]{12,}\b", " ", text)
    text = re.sub(r"\b[\w\-]{25,}\b", " ", text)
    text = re.sub(r"\b(enron|ecto|hou|mailto)\b", " ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


# ============================================================================
# FILTERING
# ============================================================================

def is_reasonable(subject: str, text: str) -> bool:
    """
    Basic quality filter for extracted messages.

    Rejects cases that are:
    - too short
    - too long 
    """
    subject = header_to_str(subject)
    text = header_to_str(text)

    combined = f"{subject} {text}".strip()
    if len(combined) < 30:
        return False

    token_count = len(re.findall(r"\b\w+\b", combined))
    return 5 <= token_count <= 2500


def has_meaningful_content(subject: str, body: str, msg, meta: dict) -> bool:
    """
    Decide whether an email has usable content, even if visible text is tiny.

    Accept if at least one of these is true:
    - normal visible text exists
    - URL exists in raw message
    - email address exists in raw message
    - IP exists in raw message
    - image/form/iframe suggests meaningful HTML content
    - certain HTML signals appear
    """
    subject = header_to_str(subject)
    body = header_to_str(body)
    combined = f"{subject} {body}".strip()

    if combined and len(re.findall(r"\w", combined)) >= 5:
        return True

    raw_msg = str(msg)
    if URL_RE.search(raw_msg):
        return True
    if EMAIL_RE.search(raw_msg):
        return True
    if IP_RE.search(raw_msg):
        return True

    if meta.get("has_embedded_images", 0):
        return True
    if meta.get("has_form_tag", 0):
        return True
    if meta.get("has_iframe_tag", 0):
        return True

    raw_lower = raw_msg.lower()
    html_signals = [
        "<a ",
        "<img",
        "<form",
        "<table",
        "<title>",
        "<body",
        "href=",
        "src=",
        "action=",
        "cid:",
    ]
    if any(signal in raw_lower for signal in html_signals):
        return True

    return False


# ============================================================================
# URL / FEATURE HELPERS
# ============================================================================

def extract_urls(raw: str, html_links: list[str] | None = None) -> list[str]:
    """
    Extract URLs from normal text plus optional links recovered directly
    from HTML attributes.
    """
    urls = [normalize_url(u) for u in URL_RE.findall(raw)]
    if html_links:
        urls.extend(normalize_url(u) for u in html_links)

    out = []
    seen = set()
    for url in urls:
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def url_has_at_symbol(url: str) -> bool:
    """
    Detect whether a URL contains '@' in the netloc.
    This is a classic obfuscation trick in phishing URLs.
    """
    try:
        parsed = urlparse(url)
        return "@" in parsed.netloc
    except Exception:
        return "@" in url


def get_domains(urls: list[str]) -> set[str]:
    """
    Return the set of distinct domains/netlocs extracted from URLs.
    """
    domains = set()
    for url in urls:
        netloc = safe_netloc(url)
        if netloc:
            domains.add(netloc)
    return domains


def count_keyword_hits(text: str) -> int:
    """
    Count how many tokens belong to the phishing keyword lexicon.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for t in tokens if t in PHISH_KEYWORDS)


def lightweight_features(subject: str, body: str, cleaned_text: str, meta: dict) -> dict:
    """
    Compute lightweight structural features.

    Important design choice:
    - raw features such as URLs, domains, punctuation, uppercase tokens
      are computed from the extracted subject/body content
    - keyword hits operate on cleaned_text
    """
    raw = f"{subject}\n{body}"
    tokens = re.findall(r"\b\w+\b", cleaned_text.lower())
    upper_tokens = re.findall(r"\b[A-Z]{2,}\b", raw)

    urls = extract_urls(raw, meta.get("html_extracted_links", []))
    domains = get_domains(urls)

    has_ip_url = int(any(IP_URL_RE.search(url) for url in urls))
    has_at_in_url = int(any(url_has_at_symbol(url) for url in urls))
    has_external_links = int(len(domains) > 0)

    return {
        "char_len": len(cleaned_text),
        "token_len": len(tokens),
        "subject_len": len(subject),
        "num_urls": len(urls),
        "num_emails": len(EMAIL_RE.findall(raw)),
        "num_domains": len(domains),
        "has_ip_url": has_ip_url,
        # "has_at_in_url": has_at_in_url,
        "has_external_links": has_external_links,
        "num_exclamation": raw.count("!"),
        "num_upper_tokens": len(upper_tokens),
        "keyword_hits": count_keyword_hits(cleaned_text),
        # "has_html": meta["has_html"],
        "has_form_tag": meta["has_form_tag"],
        "has_script_tag": meta["has_script_tag"],
        "has_iframe_tag": meta["has_iframe_tag"],
        # "has_embedded_images": meta["has_embedded_images"],
        "num_attachments": meta["num_attachments"],
    }


# ============================================================================
# DEDUPLICATION
# ============================================================================

def canonical_key(subject: str, body: str) -> str:
    """
    Create a canonical key for deduplication based on cleaned subject and body.
    - use cleaned text, not aggressively masked text
    - keep subject + cleaned body
    - normalize reply prefixes
    - trim excessive whitespace
    """
    subject = clean_text(subject).lower()
    body = clean_text(body).lower()

    subject = re.sub(r"^\s*(re|fw|fwd)\s*:\s*", "", subject)
    key = f"{subject}\n{body}"
    key = MULTISPACE_RE.sub(" ", key).strip()

    # Keep a large cap to avoid pathological memory growth but preserve more
    # content than before.
    return key[:8000]


def deduplicate(rows: list[dict]) -> tuple[list[dict], int]:
    """
    Remove near-exact duplicates using the conservative canonical key.
    """
    seen = set()
    out = []
    duplicates_removed = 0

    for row in rows:
        key = canonical_key(row["subject_raw"], row["body_raw"])
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        out.append(row)

    return out, duplicates_removed


# ============================================================================
# STATS
# ============================================================================

def build_dataset_stats(rows: list[dict]) -> dict:
    """
    Build summary statistics for numeric feature columns.
    """
    if not rows:
        return {}

    numeric_fields = [
        "char_len",
        "token_len",
        "subject_len",
        "num_urls",
        "num_emails",
        "num_domains",
        "num_exclamation",
        "num_upper_tokens",
        "keyword_hits",
        "num_attachments",
        # "has_html",
        "has_form_tag",
        "has_script_tag",
        "has_iframe_tag",
        # "has_embedded_images",
        "has_ip_url",
        # "has_at_in_url",
        "has_external_links",
    ]

    stats = {
        "num_rows": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
    }

    for field in numeric_fields:
        values = [row[field] for row in rows]
        stats[field] = {
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
            "sum": sum(values),
        }

    return stats


# ============================================================================
# MAIN PARSER
# ============================================================================

def parse_mbox(mbox_path: Path, *, label: int, source: str) -> tuple[list[dict], dict]:
    """
    Parse one mbox file into structured rows plus detailed statistics.
    """
    rows = []
    mbox = mailbox.mbox(str(mbox_path), factory=None, create=False)

    stats = {
        "total_messages_seen": 0,
        "parsed_before_dedup": 0,
        "saved_after_dedup": 0,
        "excluded_total": 0,
        "excluded_reasons": Counter(),
        "duplicates_removed": 0,
    }

    for i, msg in enumerate(mbox):
        stats["total_messages_seen"] += 1

        try:
            subject, body, meta = extract_subject_body_meta(msg)
            subject = header_to_str(subject)
            body = header_to_str(body)
        except Exception:
            stats["excluded_total"] += 1
            stats["excluded_reasons"]["parse_error"] += 1
            continue

        if not has_meaningful_content(subject, body, msg, meta):
            stats["excluded_total"] += 1
            stats["excluded_reasons"]["empty_body"] += 1
            continue

        if not is_reasonable(subject, body):
            stats["excluded_total"] += 1
            stats["excluded_reasons"]["failed_raw_reasonableness"] += 1
            continue

        # Text field intended for modeling.
        text = debias_text(f"Subject: {subject}\n\n{body}")

        if not is_reasonable(subject, text):
            stats["excluded_total"] += 1
            stats["excluded_reasons"]["failed_cleaned_reasonableness"] += 1
            continue

        row = {
            "id": f"{source}_{i}",
            "label": int(label),
            "source": source,
            "subject": subject,
            "text": text,

            # Keep raw extracted versions internally for safer deduplication.
            # They are dropped before CSV export.
            "subject_raw": subject,
            "body_raw": body,
        }

        row.update(lightweight_features(subject, body, text, meta))
        rows.append(row)

    stats["parsed_before_dedup"] = len(rows)

    deduped_rows, duplicates_removed = deduplicate(rows)
    stats["duplicates_removed"] = duplicates_removed
    stats["excluded_total"] += duplicates_removed
    stats["excluded_reasons"]["duplicate"] += duplicates_removed
    stats["saved_after_dedup"] = len(deduped_rows)

    # Drop internal helper fields before export/stats.
    export_rows = []
    for row in deduped_rows:
        row = dict(row)
        row.pop("subject_raw", None)
        row.pop("body_raw", None)
        export_rows.append(row)

    stats["dataset_stats"] = build_dataset_stats(export_rows)
    return export_rows, stats


# ============================================================================
# OUTPUT
# ============================================================================

def write_csv(rows: list[dict], out_path: Path) -> None:
    """
    Save parsed rows to CSV.
    """
    if not rows:
        raise RuntimeError("No usable mailbox rows were parsed.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_stats(stats: dict) -> None:
    """
    Pretty-print parser summary and numeric dataset statistics.
    """
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)
    print(f"Total messages seen:        {stats['total_messages_seen']}")
    print(f"Parsed before dedup:        {stats['parsed_before_dedup']}")
    print(f"Saved after dedup:          {stats['saved_after_dedup']}")
    print(f"Excluded total:             {stats['excluded_total']}")
    print(f"Duplicates removed:         {stats['duplicates_removed']}")

    print("\nExclusion reasons:")
    if stats["excluded_reasons"]:
        for reason, count in sorted(stats["excluded_reasons"].items()):
            print(f"  - {reason}: {count}")
    else:
        print("  - none")

    ds = stats.get("dataset_stats", {})
    if ds:
        print("\n" + "=" * 60)
        print("CREATED DATASET STATISTICS")
        print("=" * 60)
        print(f"Rows:                       {ds['num_rows']}")
        print(f"Label counts:               {ds['label_counts']}")
        print(f"Source counts:              {ds['source_counts']}")

        fields_to_show = [
            "char_len",
            "token_len",
            "subject_len",
            "num_urls",
            "num_emails",
            "num_domains",
            "num_exclamation",
            "num_upper_tokens",
            "keyword_hits",
            "num_attachments",
            # "has_html",
            "has_form_tag",
            "has_script_tag",
            "has_iframe_tag",
         # "has_embedded_images",
            "has_ip_url",
            # "has_at_in_url",
            "has_external_links",
        ]

        for field in fields_to_show:
            field_stats = ds[field]
            print(
                f"{field:25s} "
                f"min={field_stats['min']:<8} "
                f"max={field_stats['max']:<8} "
                f"mean={field_stats['mean']:.2f} "
                f"sum={field_stats['sum']}"
            )


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """
    Command-line entry point.

    Example:
    python parse_mailbox.py \
        --mbox data/raw/mbox/enron.mbox \
        --out data/processed/enron_parsed.csv \
        --label 0 \
        --source enron
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mbox",
        type=Path,
        required=True,
        help="Path to the input .mbox file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to output CSV",
    )
    parser.add_argument(
        "--label",
        type=int,
        required=True,
        choices=[0, 1],
        help="Dataset label: 0 for ham, 1 for phishing",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Dataset source name, e.g. enron or nazario",
    )
    args = parser.parse_args()

    rows, stats = parse_mbox(args.mbox, label=args.label, source=args.source)
    write_csv(rows, args.out)

    print(f"\nParsed {len(rows)} emails from: {args.mbox.resolve()}")
    print(f"Saved to: {args.out.resolve()}")
    print_stats(stats)


if __name__ == "__main__":
    main()