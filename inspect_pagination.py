import requests
import re

URL = "https://review.travel.rakuten.co.jp/hotel/voice/76373"

response = requests.get(
    URL,
    timeout=(10, 60),
    headers={"User-Agent": "Mozilla/5.0"}
)

response.raise_for_status()
response.encoding = "utf-8"

html = response.text

print("HTML length:", len(html))

# Look for common pagination-related terms
terms = [
    "pagination",
    "page",
    "offset",
    "limit",
    "reviewList",
    "currentPage",
    "total"
]

print("\nPagination-related occurrences:")

for term in terms:
    count = len(re.findall(term, html, re.IGNORECASE))
    print(f"{term}: {count}")

# Show URLs in the HTML that contain page/offset information
urls = re.findall(r'https?[^"\'<>\s]+', html)

print("\nPotential pagination URLs:")

found = 0

for url in urls:
    if any(x in url.lower() for x in ["page=", "offset=", "review"]):
        print(url[:300])
        found += 1

        if found >= 20:
            break

print("\nPotential pagination links found:", found)