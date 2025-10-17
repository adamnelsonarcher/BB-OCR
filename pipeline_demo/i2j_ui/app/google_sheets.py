import os
import datetime
from typing import Any, Dict, Optional

_CLIENT = None
_SHEET = None

# Target 8-column table header
_TABLE8_HEADER = [
    "Book Title",
    "Author",
    "Year",
    "Publisher",
    "Has ISBN",
    "Link Found",
    "Accept/Reject",
    "Comments",
]


def _load_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    creds_path = os.environ.get("GOOGLE_SHEETS_CREDENTIALS_JSON")
    if not creds_path:
        return None
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        return None
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        _CLIENT = gspread.authorize(creds)
        return _CLIENT
    except Exception:
        return None


def _load_sheet():
    global _SHEET
    if _SHEET is not None:
        return _SHEET
    client = _load_client()
    if client is None:
        return None
    spreadsheet_id = os.environ.get("GOOGLE_SHEETS_SPREADSHEET_ID")
    if not spreadsheet_id:
        return None
    try:
        sh = client.open_by_key(spreadsheet_id)
        worksheet_name = os.environ.get("GOOGLE_SHEETS_WORKSHEET", "ReviewLog")
        try:
            ws = sh.worksheet(worksheet_name)
        except Exception:
            ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="20")
            # Best-effort: add default header (16-col audit sheet)
            try:
                ws.append_row([
                    "timestamp_iso",
                    "stage",
                    "action",
                    "id",
                    "source_path",
                    "title",
                    "authors_csv",
                    "isbn_13",
                    "isbn_10",
                    "publisher",
                    "publication_date",
                    "pricing_provider",
                    "price_amount",
                    "price_currency",
                    "comment",
                    "error",
                ])
            except Exception:
                pass
        _SHEET = ws
        return _SHEET
    except Exception:
        return None


def is_configured() -> bool:
    return _load_sheet() is not None


def append_row(
    *,
    stage: str,
    action: str,
    id: Optional[str],
    source_path: Optional[str],
    comment: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    offer: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> bool:
    ws = _load_sheet()
    if ws is None:
        return False
    md = metadata or {}
    ts = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    title = md.get("title") if isinstance(md, dict) else None
    authors = md.get("authors") if isinstance(md, dict) else None
    if isinstance(authors, list):
        authors_csv = ", ".join([str(a) for a in authors if a is not None])
    elif isinstance(authors, str):
        authors_csv = authors
    else:
        authors_csv = None
    isbn_13 = md.get("isbn_13") if isinstance(md, dict) else None
    isbn_10 = md.get("isbn_10") if isinstance(md, dict) else None
    publisher = md.get("publisher") if isinstance(md, dict) else None
    publication_date = md.get("publication_date") if isinstance(md, dict) else None
    prov = offer.get("provider") if isinstance(offer, dict) else None
    amt = offer.get("amount") if isinstance(offer, dict) else None
    cur = offer.get("currency") if isinstance(offer, dict) else None
    # Decide format by inspecting the header row
    try:
        header_values = ws.row_values(1) if hasattr(ws, 'row_values') else []
    except Exception:
        header_values = []

    def _extract_year(v: Optional[str]) -> str:
        if not v:
            return ""
        try:
            import re
            m = re.search(r"(18|19|20)\d{2}", str(v))
            return m.group(0) if m else ""
        except Exception:
            return ""

    # 8-column table format
    if header_values and header_values[:len(_TABLE8_HEADER)] == _TABLE8_HEADER:
        has_isbn = "yes" if (isbn_13 or isbn_10) else "no"
        link_found = "yes" if (
            (isinstance(offer, dict) and (offer.get("url") or offer.get("info_url")))
            or (isinstance(md, dict) and (md.get("info_url") or md.get("source_url")))
        ) else "no"
        decision = "accept" if str(action).lower().startswith("approv") else "reject"
        try:
            ws.append_row([
                title or "",
                authors_csv or "",
                _extract_year(publication_date),
                publisher or "",
                has_isbn,
                link_found,
                decision,
                comment or "",
            ])
            return True
        except Exception:
            return False

    # Default: 16-column audit sheet
    try:
        ws.append_row([
            ts,
            stage,
            action,
            id or "",
            source_path or "",
            title or "",
            authors_csv or "",
            isbn_13 or "",
            isbn_10 or "",
            publisher or "",
            publication_date or "",
            prov or "",
            amt if (isinstance(amt, (int, float)) or (isinstance(amt, str) and amt)) else "",
            cur or "",
            comment or "",
            error or "",
        ])
        return True
    except Exception:
        return False


def connectivity() -> Dict[str, Any]:
    client = _load_client()
    if client is None:
        return {"ok": False, "error": "client_unavailable"}
    ws = _load_sheet()
    if ws is None:
        return {"ok": False, "error": "sheet_unavailable"}
    try:
        title = ws.title
        return {"ok": True, "worksheet": title}
    except Exception as e:
        return {"ok": False, "error": str(e)}


