import requests
import re
import json

u = "https://review.travel.rakuten.co.jp/hotel/voice/76373"

r = requests.get(u, timeout=(10, 60))
r.encoding = "utf-8"
html = r.text

print("HTML length:", len(html))

# Find the embedded review list
marker = '"reviewList":{"data":{'
start = html.find(marker)

print("Review JSON marker found:", start != -1)

if start == -1:
    print("Could not find reviewList.")
    raise SystemExit

# Find the contents array
contents_marker = '"contents":'
contents_start = html.find(contents_marker, start)

print("Contents marker found:", contents_start != -1)

if contents_start == -1:
    print("Could not find contents.")
    raise SystemExit

array_start = html.find("[", contents_start)

# Use JSONDecoder to read the array correctly,
# including nested objects/arrays inside each review.
decoder = json.JSONDecoder()

reviews, end_position = decoder.raw_decode(html[array_start:])

print("Reviews extracted:", len(reviews))

if reviews:
    print("\nFIRST REVIEW:\n")
    print(json.dumps(reviews[0], ensure_ascii=False, indent=2))
else:
    print("No reviews found.")