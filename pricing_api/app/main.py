import os
import asyncio
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
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


app = FastAPI(title="Pricing API (experimental)")

# Static UI mounting
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
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


