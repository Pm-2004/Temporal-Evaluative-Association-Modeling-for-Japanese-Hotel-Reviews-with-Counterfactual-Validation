import requests
import json
import time
import os


# ============================================================
# SETTINGS
# ============================================================

HOTEL_ID = "1799"

BASE_URL = f"https://review.travel.rakuten.co.jp/hotel/voice/{HOTEL_ID}"

OUTPUT_DIR = r"data\collected"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"hotel_{HOTEL_ID}_reviews.jsonl"
)

# Wait between requests.
DELAY_SECONDS = 2

# ============================================================
# IMPORTANT
# ============================================================
# For now we keep this at 5 for testing.
#
# After this test passes, change it to:
#
# MAX_PAGES = None
#
# to collect the complete hotel history.
# ============================================================

MAX_PAGES = None


# ============================================================
# GET ONE PAGE
# ============================================================

def get_page(session, page_number):

    if page_number == 1:
        url = BASE_URL
    else:
        url = f"{BASE_URL}?page={page_number}"

    print(f"\nRequesting page {page_number}...")
    print(f"URL: {url}")

    response = session.get(
        url,
        timeout=(10, 60)
    )

    response.raise_for_status()
    response.encoding = "utf-8"

    html = response.text

    print(f"HTML length: {len(html):,}")

    # --------------------------------------------------------
    # Find review JSON
    # --------------------------------------------------------

    marker = '"reviewList":{"data":{'

    start = html.find(marker)

    if start == -1:
        raise RuntimeError(
            "Could not find reviewList in page."
        )

    # --------------------------------------------------------
    # Find total review count
    # --------------------------------------------------------

    total_marker = '"total":'

    total_start = html.find(
        total_marker,
        start
    )

    total_reviews = None

    if total_start != -1:

        number_start = (
            total_start
            + len(total_marker)
        )

        number_end = html.find(
            ",",
            number_start
        )

        if number_end != -1:

            try:
                total_reviews = int(
                    html[
                        number_start:number_end
                    ]
                )
            except ValueError:
                pass

    # --------------------------------------------------------
    # Find contents array
    # --------------------------------------------------------

    contents_marker = '"contents":'

    contents_start = html.find(
        contents_marker,
        start
    )

    if contents_start == -1:
        raise RuntimeError(
            "Could not find review contents."
        )

    array_start = html.find(
        "[",
        contents_start
    )

    if array_start == -1:
        raise RuntimeError(
            "Could not find review array."
        )

    decoder = json.JSONDecoder()

    reviews, _ = decoder.raw_decode(
        html[array_start:]
    )

    return reviews, total_reviews


# ============================================================
# LOAD EXISTING REVIEWS
# ============================================================

def load_existing_reviews():

    existing = {}

    if not os.path.exists(OUTPUT_FILE):
        return existing

    print("\nExisting output file found.")
    print("Loading existing reviews...")

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile:

        for line in infile:

            line = line.strip()

            if not line:
                continue

            try:

                review = json.loads(line)

                review_id = str(
                    review.get("id", "")
                )

                if review_id:
                    existing[review_id] = review

            except json.JSONDecodeError:

                print(
                    "Warning: invalid JSON line skipped."
                )

    print(
        f"Existing unique reviews: {len(existing)}"
    )

    return existing


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load existing data
    # --------------------------------------------------------

    existing_reviews = load_existing_reviews()

    seen_ids = set(
        existing_reviews.keys()
    )

    # --------------------------------------------------------
    # HTTP session
    # --------------------------------------------------------

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        )
    })

    total_reported = None
    pages_collected = 0
    new_reviews = 0

    # --------------------------------------------------------
    # Open output in APPEND mode.
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as outfile:

        page = 1

        while True:

            # ------------------------------------------------
            # Page limit for testing
            # ------------------------------------------------

            if (
                MAX_PAGES is not None
                and page > MAX_PAGES
            ):
                break

            try:

                reviews, reported_total = get_page(
                    session,
                    page
                )

            except Exception as e:

                print(
                    f"\nERROR on page {page}: {e}"
                )

                print(
                    "Stopping safely."
                )

                break

            # ------------------------------------------------
            # Store total count
            # ------------------------------------------------

            if reported_total is not None:

                total_reported = reported_total

                print(
                    f"Hotel reported total reviews: "
                    f"{total_reported:,}"
                )

            # ------------------------------------------------
            # No reviews
            # ------------------------------------------------

            if not reviews:

                print(
                    f"No reviews found on page {page}."
                )

                break

            print(
                f"Reviews found: {len(reviews)}"
            )

            page_new = 0

            # ------------------------------------------------
            # Deduplicate
            # ------------------------------------------------

            for review in reviews:

                review_id = str(
                    review.get("id", "")
                )

                if not review_id:
                    continue

                if review_id in seen_ids:
                    continue

                seen_ids.add(review_id)

                outfile.write(
                    json.dumps(
                        review,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                page_new += 1
                new_reviews += 1

            print(
                f"New unique reviews: {page_new}"
            )

            pages_collected += 1

            # ------------------------------------------------
            # Stop if we've reached reported total
            # ------------------------------------------------

            if (
                total_reported is not None
                and len(seen_ids) >= total_reported
            ):

                print(
                    "\nReported total reached."
                )

                break

            # ------------------------------------------------
            # Final-page safety check
            # ------------------------------------------------

            if len(reviews) < 20:

                print(
                    "\nFewer than 20 reviews returned."
                )

                print(
                    "Likely final page."
                )

                break

            page += 1

            time.sleep(DELAY_SECONDS)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("COLLECTION SUMMARY")
    print("=" * 60)

    print(
        f"Hotel ID: {HOTEL_ID}"
    )

    print(
        f"Pages processed this run: "
        f"{pages_collected}"
    )

    print(
        f"New reviews this run: "
        f"{new_reviews}"
    )

    print(
        f"Total unique reviews in file: "
        f"{len(seen_ids)}"
    )

    if total_reported is not None:

        print(
            f"Hotel reported total: "
            f"{total_reported}"
        )

        remaining = (
            total_reported
            - len(seen_ids)
        )

        print(
            f"Remaining reviews: "
            f"{max(remaining, 0)}"
        )

    print(
        f"Output file: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()