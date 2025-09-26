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


def _to_float(num_str: str) -> Optional[float]:
    if not num_str:
        return None
    s = str(num_str).strip()
    # Determine decimal separator: pick the last occurrence among ',' and '.' as decimal
    last_comma = s.rfind(',')
    last_dot = s.rfind('.')
    if last_comma == -1 and last_dot == -1:
        # digits only
        try:
            return float(s)
        except Exception:
            return None
    # Identify decimal separator position
    if last_comma > last_dot:
        dec = ','
        thou = '.'
    else:
        dec = '.'
        thou = ','
    # Remove thousand separators and normalize decimal
    parts = s.replace(thou, '')
    parts = parts.replace(dec, '.')
    try:
        return float(parts)
    except Exception:
        return None


def _parse_price(text: str) -> (Optional[str], Optional[float]):
    if not text:
        return None, None
    t = re.sub(r"\s+", " ", str(text)).strip()
    # Common currency tokens and symbols
    symbol_to_ccy = {
        '$': 'USD', '£': 'GBP', '€': 'EUR',
    }
    word_to_ccy = {
        'USD': 'USD', 'US$': 'USD', 'US$': 'USD', 'US DOLLARS': 'USD',
        'GBP': 'GBP', 'EUR': 'EUR', 'CAD': 'CAD', 'AUD': 'AUD',
        'C$': 'CAD', 'CA$': 'CAD', 'AU$': 'AUD',
    }
    patterns = [
        r"\b(USD|GBP|EUR|CAD|AUD)\b\s*([0-9][0-9.,]*)",
        r"\b(US\$|C\$|CA\$|AU\$)\b\s*([0-9][0-9.,]*)",
        r"([\$£€])\s*([0-9][0-9.,]*)",
        r"([0-9][0-9.,]*)\s*\b(USD|GBP|EUR|CAD|AUD)\b",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if not m:
            continue
        if len(m.groups()) == 2:
            g1, g2 = m.group(1), m.group(2)
            if g1 in symbol_to_ccy:
                ccy = symbol_to_ccy[g1]
                amt = _to_float(g2)
                return ccy, amt
            if g2 in word_to_ccy or g2.upper() in word_to_ccy:
                ccy = word_to_ccy.get(g2, word_to_ccy.get(g2.upper(), None))
                amt = _to_float(g1)
                return ccy, amt
            ccy = word_to_ccy.get(g1, word_to_ccy.get(g1.upper(), None))
            amt = _to_float(g2)
            if ccy or amt is not None:
                return ccy, amt
    # Fallback: any amount with symbol adjacent (e.g., US$12.34)
    m = re.search(r"(US\$|C\$|CA\$|AU\$)([0-9][0-9.,]*)", t, flags=re.IGNORECASE)
    if m:
        ccy = word_to_ccy.get(m.group(1).upper(), None)
        amt = _to_float(m.group(2))
        return ccy, amt
    return None, None


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
                # Price extraction: check multiple potential selectors and attributes
                price_el = (c.select_one("[itemprop='price']") or
                            c.select_one("meta[itemprop='price']") or
                            c.select_one(".item-price") or
                            c.select_one(".srp-item-price") or
                            c.select_one(".price") or
                            c.select_one("[data-cy='listing-price']") or
                            c.select_one("[data-cy='item-price']"))
                price_text = None
                currency = None
                amount = None
                if price_el:
                    # meta or span with content attribute
                    if price_el.has_attr("content"):
                        amount = _to_float(price_el.get("content"))
                        # look for explicit currency near
                        cur_el = c.select_one("meta[itemprop='priceCurrency']") or c.select_one("[itemprop='priceCurrency']")
                        if cur_el and cur_el.has_attr("content"):
                            currency = (cur_el.get("content") or "").strip().upper() or None
                    if amount is None:
                        price_text = price_el.get_text(" ", strip=True)
                if amount is None and (price_text or c):
                    ccy, amt = _parse_price(price_text or c.get_text(" ", strip=True))
                    currency = currency or ccy
                    amount = amount or amt
                pub_el = c.select_one(".publisher, .pub, .text-muted")
                pub_text = pub_el.get_text(strip=True) if pub_el else None

                if not title_text and not href and not price_text:
                    continue

                # If still missing, leave currency/amount as None

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



