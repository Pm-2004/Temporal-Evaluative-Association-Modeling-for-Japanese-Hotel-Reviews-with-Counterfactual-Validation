import requests
import json

URL = "https://review.travel.rakuten.co.jp/hotel/voice/76373?page=2"

response = requests.get(
    URL,
    timeout=(10, 60),
    headers={"User-Agent": "Mozilla/5.0"}
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

decoder = json.JSONDecoder()
reviews, _ = decoder.raw_decode(html[array_start:])

print("Reviews extracted:", len(reviews))

if reviews:
    print("\nFIRST REVIEW ON PAGE 2")
    print("----------------------")
    print("Review ID:", reviews[0].get("id"))
    print("Date:", reviews[0].get("postDateTime"))
    print("Title:", reviews[0].get("title"))

    print("\nLAST REVIEW ON PAGE 2")
    print("---------------------")
    print("Review ID:", reviews[-1].get("id"))
    print("Date:", reviews[-1].get("postDateTime"))
    print("Title:", reviews[-1].get("title"))