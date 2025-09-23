import os
import asyncio
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pricing_api.core.aggregator import aggregate_offers, DEFAULT_PROVIDERS


class LookupRequest(BaseModel):
    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: Optional[List[str]] = None
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    isbn_13: Optional[str] = None
    isbn_10: Optional[str] = None
    providers: Optional[List[str]] = None


class LookupResponse(BaseModel):
    query: Dict[str, Any]
    providers: List[str]
    offers: List[Dict[str, Any]]
    errors: Dict[str, str] = {}


app = FastAPI(title="Pricing API (pipeline_demo)")

# Static UI mounting
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return {"status": "ok", "providers": [name for name, _ in DEFAULT_PROVIDERS]}


@app.get("/ui")
async def ui():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        return {"message": "UI not found", "static_dir": STATIC_DIR}
    return FileResponse(index_path)


@app.get("/providers")
async def providers():
    return {"providers": [name for name, _ in DEFAULT_PROVIDERS]}


def _processed_dirs() -> List[str]:
    # Look for outputs inside pipeline_demo/extractor
    candidates = [
        os.path.join(REPO_ROOT, "extractor", "output"),
        os.path.join(REPO_ROOT, "extractor", "batch_output"),
    ]
    return [d for d in candidates if os.path.isdir(d)]


@app.get("/processed/list")
async def processed_list():
    items = []
    for base in _processed_dirs():
        for name in os.listdir(base):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(base, name)
            try:
                st = os.stat(path)
            except Exception:
                continue
            label = name
            items.append({
                "label": label,
                "path": path,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "dir": base,
            })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"items": items}


def _allowed_path(p: str) -> bool:
    try:
        rp = os.path.abspath(p)
        for base in _processed_dirs():
            if rp.startswith(os.path.abspath(base) + os.sep) or rp == os.path.abspath(base):
                return True
        return False
    except Exception:
        return False


@app.get("/processed/load")
async def processed_load(path: str):
    if not _allowed_path(path):
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    payload = {
        "title": data.get("title"),
        "subtitle": data.get("subtitle"),
        "authors": data.get("authors"),
        "publisher": data.get("publisher"),
        "publication_date": data.get("publication_date"),
        "isbn_13": data.get("isbn_13"),
        "isbn_10": data.get("isbn_10"),
    }
    return {"path": path, "payload": payload, "raw": data}


@app.post("/lookup", response_model=LookupResponse)
async def lookup(req: LookupRequest):
    offers, errors = await aggregate_offers(
        title=req.title,
        authors=req.authors or [],
        isbn_13=req.isbn_13,
        isbn_10=req.isbn_10,
        publisher=req.publisher,
        publication_date=req.publication_date,
        providers=req.providers,  # None => default list
        timeout_seconds=8.0,
    )
    return LookupResponse(
        query=req.model_dump(),
        providers=[name for name, _ in DEFAULT_PROVIDERS] if req.providers is None else req.providers,
        offers=offers,
        errors=errors,
    )


