import os
from typing import Optional, List, Dict, Any
import httpx


class GoogleBooksProvider:
    BASE = "https://www.googleapis.com/books/v1/volumes"

    async def lookup(
        self,
        *,
        title: Optional[str],
        authors: List[str],
        isbn_13: Optional[str],
        isbn_10: Optional[str],
        publisher: Optional[str],
        publication_date: Optional[str],
    ) -> List[Dict[str, Any]]:
        key = os.getenv("GOOGLE_BOOKS_API_KEY")
        q_parts = []
        if isbn_13:
            q_parts.append(f"isbn:{isbn_13}")
        if isbn_10:
            q_parts.append(f"isbn:{isbn_10}")
        if title:
            q_parts.append(f"intitle:{title}")
        for a in authors or []:
            if a:
                q_parts.append(f"inauthor:{a}")
        if publisher:
            q_parts.append(f"inpublisher:{publisher}")
        q = "+".join(part.replace(" ", "+") for part in q_parts) or title or ""
        params = {"q": q, "maxResults": 5}
        if key:
            params["key"] = key
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            data = r.json()
        items = data.get("items", []) or []
        offers: List[Dict[str, Any]] = []
        for it in items:
            vi = it.get("volumeInfo", {})
            identifiers = vi.get("industryIdentifiers", [])
            isbn13 = next((i.get("identifier") for i in identifiers if i.get("type") == "ISBN_13"), None)
            isbn10 = next((i.get("identifier") for i in identifiers if i.get("type") == "ISBN_10"), None)
            offers.append({
                "provider": "google_books",
                "listing_id": it.get("id"),
                "title": vi.get("title"),
                "authors": vi.get("authors", []),
                "publisher": vi.get("publisher"),
                "publication_date": vi.get("publishedDate"),
                "description": vi.get("description"),
                "page_count": vi.get("pageCount"),
                "categories": vi.get("categories"),
                "language": vi.get("language"),
                "isbn_13": isbn13,
                "isbn_10": isbn10,
                "currency": None,
                "amount": None,
                "url": vi.get("infoLink") or it.get("selfLink"),
                "source": "metadata",
            })
        return offers


