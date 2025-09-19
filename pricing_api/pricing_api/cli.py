import asyncio
import argparse
from typing import List

from pricing_api.core.aggregator import aggregate_offers


def main():
    parser = argparse.ArgumentParser(description="Test pricing providers")
    parser.add_argument("--title", type=str)
    parser.add_argument("--author", action="append", default=[])
    parser.add_argument("--isbn13", type=str)
    parser.add_argument("--isbn10", type=str)
    parser.add_argument("--providers", nargs="+", default=None)
    args = parser.parse_args()

    async def run():
        offers, errors = await aggregate_offers(
            title=args.title,
            authors=args.author,
            isbn_13=args.isbn13,
            isbn_10=args.isbn10,
            publisher=None,
            publication_date=None,
            providers=args.providers,
            timeout_seconds=8.0,
        )
        print("Offers:")
        for o in offers:
            print(o)
        if errors:
            print("Errors:")
            for k, v in errors.items():
                print(k, ":", v)

    asyncio.run(run())


if __name__ == "__main__":
    main()


