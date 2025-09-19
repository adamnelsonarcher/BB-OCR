from typing import Optional, List, Dict, Any


class AbeBooksStubProvider:
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
        # Placeholder: integrate AbeBooks API here
        return []


