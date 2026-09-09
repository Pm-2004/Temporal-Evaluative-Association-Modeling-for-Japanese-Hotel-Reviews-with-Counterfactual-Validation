import json
import glob
import os
from datetime import datetime
from pathlib import Path



# ============================================================
# PATHS
# ============================================================

INPUT_PATTERN = r"data\collected\hotel_*_reviews.jsonl"

OUTPUT_DIR = r"data\processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "research_corpus.jsonl")


# ============================================================
# HELPERS
# ============================================================

def extract_hotel_id(filepath):
    """
    Extract hotel ID from:

    data\collected\hotel_108080_reviews.jsonl

    Result:
    108080
    """

    filename = os.path.basename(filepath)

    # _108080_reviews.jsonl
    prefix = "_"
    suffix = "_reviews.jsonl"

    if filename.startswith(prefix) and filename.endswith(suffix):
        hotel_id = filename[len(prefix):-len(suffix)]
        return hotel_id

    return None

def clean_text(text):
    if not text:
        return ""

    return " ".join(text.split())


def extract_scores(scores):
    """
    Convert Rakuten score list into a simple dictionary.
    """

    result = {}

    if not scores:
        return result

    for item in scores:
        code = item.get("code")
        score = item.get("score")

        if code:
            result[code.lower()] = score

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(glob.glob(INPUT_PATTERN))

    print("=" * 60)
    print("BUILDING RESEARCH CORPUS")
    print("=" * 60)

    print(f"Input pattern: {INPUT_PATTERN}")
    print(f"Hotel files found: {len(files)}")

    if not files:
        print("\nERROR: No hotel datasets found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_reviews = 0
    written_reviews = 0
    skipped_reviews = 0

    hotel_counts = {}

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

        for filepath in files:

            hotel_id = extract_hotel_id(filepath)

            hotel_count = 0

            print(f"\nProcessing hotel: {hotel_id}")
            print(f"File: {filepath}")

            with open(filepath, "r", encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    total_reviews += 1

                    try:
                        review = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_reviews += 1
                        continue

                    provider = review.get("provider") or {}
                    reservation = review.get("reservation") or {}

                    record = {
                        # ------------------------------
                        # IDENTIFICATION
                        # ------------------------------
                        "hotel_id": hotel_id,
                        "review_id": review.get("id"),

                        # ------------------------------
                        # HOTEL
                        # ------------------------------
                        "hotel_name": provider.get("name"),

                        # ------------------------------
                        # REVIEW
                        # ------------------------------
                        "title": clean_text(
                            review.get("title")
                        ),

                        "comment": clean_text(
                            review.get("comment")
                        ),

                        "overall_score": review.get(
                            "overallScore"
                        ),

                        # ------------------------------
                        # TEMPORAL INFORMATION
                        # ------------------------------
                        "post_datetime": review.get(
                            "postDateTime"
                        ),

                        "check_in_date": reservation.get(
                            "checkInDate"
                        ),

                        # ------------------------------
                        # ASPECT SCORES
                        # ------------------------------
                        "scores": extract_scores(
                            review.get("scores")
                        ),

                        # ------------------------------
                        # SOURCE
                        # ------------------------------
                        "source": "rakuten_travel"
                    }

                    out.write(
                        json.dumps(
                            record,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    written_reviews += 1
                    hotel_count += 1

            hotel_counts[hotel_id] = hotel_count

            print(f"Reviews processed: {hotel_count}")

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("RESEARCH CORPUS COMPLETE")
    print("=" * 60)

    print(f"Hotel files: {len(files)}")
    print(f"Input reviews: {total_reviews}")
    print(f"Written reviews: {written_reviews}")
    print(f"Skipped reviews: {skipped_reviews}")

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    print("\nReviews per hotel:")

    for hotel_id, count in hotel_counts.items():
        print(f"  {hotel_id}: {count}")


if __name__ == "__main__":
    main()