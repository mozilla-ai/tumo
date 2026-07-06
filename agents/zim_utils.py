"""Pure Python functions to open and search ZIM files using libzim.

Uses the libzim Python bindings directly — no MCP needed.

Dependencies (add to pyproject.toml):
    libzim
    markdownify  # already present in tumo-agents

Example:
    from zim_utils import open_zim, search_zim, get_zim_entry, list_zim_files

    results = search_zim("/path/to/wikipedia.zim", "Denny Vrandecic", limit=5)
    for r in results:
        print(r["title"])

    content = get_zim_entry("/path/to/wikipedia.zim", results[0]["path"])
    print(content[:500])
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from libzim.reader import Archive
from libzim.search import Query, Searcher

try:
    from markdownify import markdownify
except ImportError:
    markdownify = None


def _html_to_text(html: str, max_lines: int = 0) -> str:
    """Convert HTML to readable plain text.

    Uses markdownify if available, otherwise strips tags with regex.
    Strips inline CSS style attributes and other noise.
    """
    # Remove script and style blocks
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip inline style="..." attributes (major source of Wikipedia noise)
    html = re.sub(r'\sstyle="[^"]*"', "", html, flags=re.IGNORECASE)
    # Strip class and data-* attributes
    html = re.sub(r'\s(class|data-[\w-]+)="[^"]*"', "", html, flags=re.IGNORECASE)

    if markdownify:
        text = markdownify(
            html,
            heading_style="ATX",
            strip=["script", "style", "meta", "link", "head", "footer", "noscript"],
        )
    else:
        # Fallback: simple regex strip
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&#\w+;", "", text)

    # Collapse whitespace and blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if max_lines:
        lines = text.split("\n")[:max_lines]
        text = "\n".join(lines)

    return text


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def list_zim_files(*directories: str, recursive: bool = True) -> list[dict]:
    """List all .zim files under the given directories.

    Args:
        directories: One or more root directories to scan.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of dicts with keys: name, path, size_mb, modified.
    """
    results: list[dict] = []
    for dir_path in directories:
        path = Path(dir_path).expanduser().resolve()
        if not path.is_dir():
            continue
        pattern = "**/*.zim" if recursive else "*.zim"
        for file_path in path.glob(pattern):
            if file_path.is_file():
                stat = file_path.stat()
                from datetime import datetime

                results.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
    return results


# ---------------------------------------------------------------------------
# Opening
# ---------------------------------------------------------------------------

def open_zim(path: str | Path) -> Archive:
    """Open a ZIM archive for reading.

    Args:
        path: Path to the .zim file.

    Returns:
        A libzim Archive object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid ZIM archive.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"ZIM file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix.lower() != ".zim":
        raise ValueError(f"Not a ZIM file (missing .zim extension): {path}")
    return Archive(str(path))


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def get_zim_info(archive: Archive) -> dict:
    """Get metadata about an opened ZIM archive.

    Args:
        archive: An open Archive object (from open_zim).

    Returns:
        Dict with keys: filename, filesize, entry_count, article_count,
        media_count, has_fulltext_index, has_title_index, metadata_keys,
        name, language, creator, date, description (from ZIM metadata).
    """
    info: dict = {
        "filename": str(archive.filename),
        "filesize": archive.filesize,
        "filesize_mb": round(archive.filesize / (1024 * 1024), 2),
        "entry_count": archive.entry_count,
        "article_count": archive.article_count,
        "media_count": archive.media_count,
        "has_fulltext_index": archive.has_fulltext_index,
        "has_title_index": archive.has_title_index,
        "metadata_keys": archive.metadata_keys,
    }

    # Common metadata keys
    for key in ("Name", "Language", "Creator", "Date", "Description", "Publisher"):
        try:
            info[key.lower()] = archive.get_metadata(key).decode("utf-8", errors="replace")
        except Exception:
            info[key.lower()] = None

    return info


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_zim(
    path: str | Path | Archive,
    query: str,
    limit: int = 10,
    offset: int = 0,
    snippet_length: int = 1000,
) -> list[dict]:
    """Search for text within a ZIM file.

    Args:
        path: Path to the .zim file, or an already-open Archive.
        query: The search query string.
        limit: Maximum number of results to return.
        offset: Number of results to skip (for pagination).
        snippet_length: Max characters for the content snippet (0 = no limit).

    Returns:
        List of dicts with keys: path, title, snippet.

    Raises:
        ValueError: If the archive has no full-text index.
    """
    if isinstance(path, Archive):
        archive = path
        should_close = False
    else:
        archive = open_zim(path)
        should_close = True

    try:
        if not archive.has_fulltext_index:
            raise ValueError(f"Archive has no full-text index: {archive.filename}")

        query_obj = Query().set_query(query)
        searcher = Searcher(archive)
        search = searcher.search(query_obj)
        total = search.getEstimatedMatches()

        if total == 0:
            return []

        result_count = min(limit, total - offset)
        result_entries = list(search.getResults(offset, result_count))

        results = []
        for entry_id in result_entries:
            try:
                entry = archive.get_entry_by_path(entry_id)
                title = entry.title or "Untitled"
                snippet = ""

                if not entry.is_redirect:
                    try:
                        item = entry.get_item()
                        if item.mimetype.startswith("text/"):
                            raw = bytes(item.content).decode("utf-8", errors="replace")
                            if item.mimetype.startswith("text/html"):
                                snippet = _html_to_text(raw)
                            else:
                                snippet = raw
                        elif item.mimetype.startswith("image/"):
                            snippet = "[image content]"
                        else:
                            snippet = f"[{item.mimetype} content]"

                        if snippet_length and len(snippet) > snippet_length:
                            snippet = snippet[:snippet_length].strip() + "..."
                    except Exception:
                        snippet = "[unable to read content]"

                results.append({
                    "path": entry_id,
                    "title": title,
                    "snippet": snippet,
                })
            except Exception:
                results.append({
                    "path": entry_id,
                    "title": "[error reading entry]",
                    "snippet": "",
                })

        return results
    finally:
        if should_close:
            del archive


# ---------------------------------------------------------------------------
# Entry retrieval
# ---------------------------------------------------------------------------

def get_zim_entry(
    path: str | Path | Archive,
    entry_path: str,
    max_length: int = 0,
) -> dict:
    """Get the full content of a specific ZIM entry.

    Args:
        path: Path to the .zim file, or an already-open Archive.
        entry_path: The entry's internal path (e.g. "A/Some_Article").
        max_length: Max characters to return (0 = no limit).

    Returns:
        Dict with keys: path, title, mimetype, content, truncated.
    """
    if isinstance(path, Archive):
        archive = path
        should_close = False
    else:
        archive = open_zim(path)
        should_close = True

    try:
        entry = archive.get_entry_by_path(entry_path)

        # Follow redirects
        while entry.is_redirect:
            entry = entry.get_redirect_entry()

        title = entry.title or "Untitled"
        content = ""
        mimetype = ""

        try:
            item = entry.get_item()
            mimetype = item.mimetype or ""

            if mimetype.startswith("text/html"):
                raw = bytes(item.content).decode("utf-8", errors="replace")
                content = _html_to_text(raw)
            elif mimetype.startswith("text/"):
                content = bytes(item.content).decode("utf-8", errors="replace")
            elif mimetype.startswith("image/"):
                content = "[image content — cannot display directly]"
            else:
                content = f"[{mimetype} content — cannot display directly]"
        except Exception as e:
            content = f"[error retrieving content: {e}]"

        truncated = False
        if max_length and len(content) > max_length:
            content = content[:max_length] + f"\n\n... [truncated, {len(content)} chars total, showing first {max_length}]"
            truncated = True

        return {
            "path": entry_path,
            "title": title,
            "mimetype": mimetype,
            "content": content,
            "truncated": truncated,
        }
    finally:
        if should_close:
            del archive


# ---------------------------------------------------------------------------
# Browse / iterate
# ---------------------------------------------------------------------------

def iterate_zim_entries(
    path: str | Path,
    prefix: str = "",
    max_entries: int = 0,
) -> Generator[dict, None, None]:
    """Iterate over entries in a ZIM archive by index.

    Args:
        path: Path to the .zim file.
        prefix: Only yield entries whose path starts with this prefix.
        max_entries: Stop after this many entries (0 = no limit).

    Yields:
        Dicts with keys: index, path, title, is_redirect.
    """
    archive = open_zim(path)
    count = archive.entry_count
    limit = max_entries if max_entries > 0 else count

    yielded = 0
    for i in range(limit):
        try:
            entry = archive._get_entry_by_id(i)
            if prefix and not entry.path.startswith(prefix):
                continue
            yield {
                "index": i,
                "path": entry.path,
                "title": entry.title,
                "is_redirect": entry.is_redirect,
            }
            yielded += 1
            if max_entries and yielded >= max_entries:
                break
        except Exception:
            continue

    del archive


# ---------------------------------------------------------------------------
# Convenience: search and read in one call
# ---------------------------------------------------------------------------

def search_and_read(
    path: str | Path,
    query: str,
    rank: int = 0,
    max_length: int = 0,
) -> dict | None:
    """Search a ZIM file and return the content of the top result.

    Args:
        path: Path to the .zim file.
        query: The search query string.
        rank: Which result to read (0 = first/ best match).
        max_length: Max content length (0 = no limit).

    Returns:
        Dict with keys: path, title, content, truncated — or None if no matches.
    """
    results = search_zim(path, query, limit=rank + 1)
    if not results:
        return None
    entry_path = results[rank]["path"]
    return get_zim_entry(path, entry_path, max_length=max_length)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    """Quick CLI for testing: `python -m zim_utils <zim_path> <query>`"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m zim_utils <zim_file> <query>")
        sys.exit(1)

    zim_path = sys.argv[1]
    query = " ".join(sys.argv[2:])

    results = search_zim(zim_path, query, limit=5)
    if not results:
        print(f'No results for "{query}"')
        return

    print(f"Top results for \"{query}\":\n")
    for r in results:
        print(f"  {r['title']}")
        print(f"  path: {r['path']}")
        print(f"  snippet: {r['snippet'][:200]}...")
        print()

    # Show full content of top result
    if results:
        full = get_zim_entry(zim_path, results[0]["path"], max_length=3000)
        print("=== Full content of top result ===")
        print(full["content"])


if __name__ == "__main__":
    _cli()
