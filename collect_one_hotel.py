import requests
import json
from pathlib import Path


URL = "https://review.travel.rakuten.co.jp/hotel/voice/76373"


def get_reviews(url):
    response = requests.get(
        url,
        timeout=(10, 60),
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    response.raise_for_status()
    response.encoding = "utf-8"

    html = response.text

    marker = '"reviewList":{"data":{'
    start = html.find(marker)

    if start == -1:
        raise RuntimeError("reviewList not found")

    contents_marker = '"contents":'
    contents_start = html.find(contents_marker, start)

    if contents_start == -1:
        raise RuntimeError("contents not found")

    array_start = html.find("[", contents_start)

    if array_start == -1:
        raise RuntimeError("review array not found")

    decoder = json.JSONDecoder()

    reviews, _ = decoder.raw_decode(html[array_start:])

    return reviews


reviews = get_reviews(URL)

print("Reviews collected:", len(reviews))

# Save raw extracted JSON
output_dir = Path("data") / "rakuten_raw"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "hotel_76373_reviews.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        reviews,
        f,
        ensure_ascii=False,
        indent=2
    )

print("Saved to:", output_file)

if reviews:
    first = reviews[0]

    print("\nFirst review:")
    print("Review ID:", first.get("id"))
    print("Hotel:", first.get("provider", {}).get("name"))
    print("Date:", first.get("postDateTime"))
    print("Overall:", first.get("overallScore"))
    print("Title:", first.get("title"))
    print("Text:", first.get("comment"))