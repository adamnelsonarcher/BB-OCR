import asyncio
from typing import Optional, List, Dict, Tuple, Any

from pricing_api.providers.google_books import GoogleBooksProvider
from pricing_api.providers.amazon_stub import AmazonStubProvider
from pricing_api.providers.abebooks_html import AbeBooksHtmlProvider
from pricing_api.providers.biblio_stub import BiblioStubProvider


DEFAULT_PROVIDERS = [
    ("google_books", GoogleBooksProvider),
    ("amazon", AmazonStubProvider),
    ("abebooks", AbeBooksHtmlProvider),
    ("biblio", BiblioStubProvider),
]


async def aggregate_offers(
    *,
    title: Optional[str],
    authors: List[str],
    isbn_13: Optional[str],
    isbn_10: Optional[str],
    publisher: Optional[str],
    publication_date: Optional[str],
    providers: Optional[List[str]] = None,
    timeout_seconds: float = 8.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    tasks = []
    chosen = [p for p in DEFAULT_PROVIDERS if (providers is None or p[0] in providers)]
    for name, klass in chosen:
        prov = klass()
        task = prov.lookup(
            title=title,
            authors=authors,
            isbn_13=isbn_13,
            isbn_10=isbn_10,
            publisher=publisher,
            publication_date=publication_date,
        )
        tasks.append((name, task))

    offers: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}

    async def run_with_name(name: str, coro):
        try:
            return name, await asyncio.wait_for(coro, timeout=timeout_seconds)
        except Exception as e:
            return name, e

    results = await asyncio.gather(*(run_with_name(n, t) for n, t in tasks))
    for name, result in results:
        if isinstance(result, Exception):
            errors[name] = str(result)
        else:
            for o in result:
                o.setdefault("provider", name)
                offers.append(o)

    # Basic de-duplication by provider id + isbn if available
    seen = set()
    unique: List[Dict[str, Any]] = []
    for o in offers:
        key = (o.get("provider"), o.get("listing_id"), o.get("isbn_13"), o.get("isbn_10"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(o)

    return unique, errors


