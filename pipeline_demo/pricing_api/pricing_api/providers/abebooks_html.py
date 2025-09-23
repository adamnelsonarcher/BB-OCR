import re
from typing import Optional, List, Dict, Any
import httpx
from bs4 import BeautifulSoup


def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()


def _extract_year(text: str) -> Optional[str]:
    m = re.search(r"(18|19|20)\d{2}", text or "")
    return m.group(0) if m else None


class AbeBooksHtmlProvider:
    BASE = "https://www.abebooks.com/servlet/SearchResults"

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
        primary_author = authors[0] if authors else None
        year = _extract_year(publication_date or "")
        parts = [p for p in [title, primary_author, year] if p]
        if not parts and not title:
            return []
        keywords = " ".join(parts)

        params = {
            "kn": keywords,
            "sortby": "17",  # price ascending
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }

        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            r = await client.get(self.BASE, params=params)
            r.raise_for_status()
            html = r.text

        soup = BeautifulSoup(html, "html.parser")

        results: List[Dict[str, Any]] = []
        cards = soup.select(".cf .result, .srp-item, .result, .cf.search-result, .search-result")
        if not cards:
            cards = soup.select("li, div")

        q_title = _norm(title)
        q_author = _norm(primary_author)
        q_year = year

        def score_offer(o: Dict[str, Any]) -> float:
            s = 0.0
            if q_title and _norm(o.get("title")) == q_title:
                s += 3.0
            elif q_title and q_title in _norm(o.get("title")):
                s += 1.5
            if q_author and q_author in _norm(o.get("authors", [None])[0] or ""):
                s += 1.5
            if q_year and o.get("publication_date"):
                oy = _extract_year(o.get("publication_date") or "")
                if oy == q_year:
                    s += 1.0
            return s

        seen = set()
        for c in cards:
            try:
                a = c.select_one("a[title], a[href*='BookDetailsPL']") or c.find("a")
                href = a["href"] if a and a.has_attr("href") else None
                title_text = a.get("title") or (a.get_text(strip=True) if a else None)
                author_el = c.select_one(".author, .srp-author, .result-author, .text-muted")
                author_text = author_el.get_text(strip=True) if author_el else None
                price_el = c.select_one(".item-price, .srp-item-price, .price, [itemprop='price']")
                price_text = price_el.get_text(strip=True) if price_el else None
                pub_el = c.select_one(".publisher, .pub, .text-muted")
                pub_text = pub_el.get_text(strip=True) if pub_el else None

                if not title_text and not href and not price_text:
                    continue

                amount = None
                currency = None
                if price_text:
                    m = re.search(r"([A-Z]{3}|\$|£|€)\s*([0-9]+(?:[.,][0-9]{2})?)", price_text)
                    if m:
                        cur, amt = m.group(1), m.group(2)
                        currency = {"$": "USD", "£": "GBP", "€": "EUR"}.get(cur, cur)
                        try:
                            amount = float(amt.replace(",", ""))
                        except Exception:
                            amount = None

                offer = {
                    "provider": "abebooks",
                    "listing_id": href,
                    "title": title_text,
                    "authors": [author_text] if author_text else [],
                    "publisher": None,
                    "publication_date": pub_text,
                    "isbn_13": None,
                    "isbn_10": None,
                    "currency": currency,
                    "amount": amount,
                    "url": href if href and href.startswith("http") else ("https://www.abebooks.com" + href if href else None),
                    "source": "scrape",
                }
                key = (offer["url"], offer["title"], offer["amount"]) 
                if key in seen:
                    continue
                seen.add(key)
                offer["score"] = score_offer(offer)
                results.append(offer)
            except Exception:
                continue

        results.sort(key=lambda x: (x.get("score", 0.0), -(x.get("amount") or 0)), reverse=True)
        return results[:10]



