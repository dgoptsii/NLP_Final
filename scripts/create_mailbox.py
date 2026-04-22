#!/usr/bin/env python3
"""
create_mailbox.py

Convert the Enron maildir-style corpus into a single .mbox file.

Why this file exists
--------------------
The Enron corpus is commonly distributed as a directory tree of raw email files.
For reuse with a single parser, it is convenient to first convert those emails
into one mbox file. This mirrors the same project idea used in the reference
GitHub script: first create an mbox from Enron, then parse the mbox with one
shared pipeline for Enron and Nazario.

Usage
-----
# Convert only inbox-like Enron messages (recommended)
python create_mailbox.py \
    --input /path/to/maildir \
    --output /path/to/enron.mbox

# Keep all folders, not just inbox
python create_mailbox.py \
    --input /path/to/maildir \
    --output /path/to/enron_all.mbox \
    --include-all-folders

# Limit how many messages are written
python create_mailbox.py \
    --input /path/to/maildir \
    --output /path/to/enron_sample.mbox \
    --max-messages 5000

# Example with specific paths:
create_mailbox.py \
    --input /data/raw/maildir \
    --output /data/raw/mbox/enron_sample.mbox \
    --max-messages 5000
"""

from __future__ import annotations

import argparse
import mailbox
from email import policy
from email.parser import BytesParser
from pathlib import Path


EXCLUDE_FOLDER_PARTS = {
    "sent",
    "sent_items",
    "sent items",
    "_sent_mail",
    "sent_mail",
    "deleted_items",
    "deleted items",
    "drafts",
    "trash",
    "junk",
    "spam",
    "all_documents",
    "all documents",
    "calendar",
    "contacts",
    "discussion_threads",
    "notes",
    "notes_inbox",
    "outbox",
    "archive",
}


def is_enron_inbox_file(path: Path) -> bool:
    """
    Return True only for inbox-like Enron files.

    This matches the usual ham-selection logic used in Enron parsing: keep
    received mail from inbox folders and skip sent/drafts/trash/etc.
    """
    parts = [p.lower() for p in path.parts]

    if "maildir" not in parts:
        return False
    if "inbox" not in parts:
        return False
    if any(p in EXCLUDE_FOLDER_PARTS for p in parts):
        return False
    return True


def iter_enron_files(root: Path, inbox_only: bool = True):
    """Yield candidate raw Enron email files in deterministic order."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if inbox_only and not is_enron_inbox_file(path):
            continue
        yield path


def convert_enron_to_mbox(
    input_root: Path,
    output_mbox: Path,
    *,
    inbox_only: bool = True,
    max_messages: int | None = None,
) -> int:
    """
    Convert raw Enron files into one mbox file.

    Each raw file is parsed as an email message and then appended to the target
    mbox. The output mbox is overwritten if it already exists.
    """
    input_root = input_root.resolve()
    output_mbox = output_mbox.resolve()
    output_mbox.parent.mkdir(parents=True, exist_ok=True)

    if output_mbox.exists():
        output_mbox.unlink()

    mbox = mailbox.mbox(str(output_mbox), create=True)
    written = 0

    try:
        mbox.lock()
        for path in iter_enron_files(input_root, inbox_only=inbox_only):
            try:
                with open(path, "rb") as f:
                    msg = BytesParser(policy=policy.default).parse(f)
            except Exception:
                continue

            try:
                mbox.add(msg)
                written += 1
            except Exception:
                continue

            if max_messages is not None and written >= max_messages:
                break

        mbox.flush()
    finally:
        try:
            mbox.unlock()
        finally:
            mbox.close()

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the extracted Enron maildir root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .mbox file",
    )
    parser.add_argument(
        "--include-all-folders",
        action="store_true",
        help="Include all folders instead of only inbox-like mail",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Optional cap on the number of written messages",
    )
    args = parser.parse_args()

    written = convert_enron_to_mbox(
        args.input,
        args.output,
        inbox_only=not args.include_all_folders,
        max_messages=args.max_messages,
    )

    print(f"Wrote {written} messages to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
