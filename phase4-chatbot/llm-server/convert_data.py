# ============================================
# convert_data.py
# Converts all-data.csv → clean stocks.txt
# Run from your llm-server folder:
#   python convert_data.py
# ============================================

import csv
import re
import os

INPUT_FILE  = "data/all-data.csv"
OUTPUT_FILE = "data/stocks.txt"


def clean_sentence(text):
    """
    Clean a single sentence:
    - Strip leading/trailing whitespace
    - Remove multiple spaces
    - Fix common encoding issues
    - Skip sentences that are too short or too long
    """
    # Fix encoding artifacts
    text = text.encode('utf-8', errors='ignore').decode('utf-8')

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Skip very short or very long sentences
    if len(text) < 30 or len(text) > 500:
        return None

    # Make sure it ends with punctuation
    if text[-1] not in '.!?':
        text = text + '.'

    return text


def convert():
    sentences  = []
    skipped    = 0
    duplicates = 0
    seen       = set()

    print(f"Reading {INPUT_FILE}...")

    with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                skipped += 1
                continue

            # Column 0 = sentiment label (positive/negative/neutral)
            # Column 1 = the actual sentence
            sentiment = row[0].strip().lower()
            sentence  = row[1].strip()

            # Clean the sentence
            cleaned = clean_sentence(sentence)
            if not cleaned:
                skipped += 1
                continue

            # Skip duplicates
            if cleaned in seen:
                duplicates += 1
                continue

            seen.add(cleaned)
            sentences.append(cleaned)

    # Write to stocks.txt — one sentence per line
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write('\n'.join(sentences))

    # Stats
    total_chars = sum(len(s) for s in sentences)
    print(f"\n✅ Conversion complete!")
    print(f"   Sentences written:  {len(sentences):,}")
    print(f"   Sentences skipped:  {skipped:,}")
    print(f"   Duplicates removed: {duplicates:,}")
    print(f"   Total characters:   {total_chars:,}")
    print(f"   Output file:        {OUTPUT_FILE}")
    print(f"\n📊 Data size comparison:")
    print(f"   Before: ~3,500 characters")
    print(f"   After:  {total_chars:,} characters  ({total_chars//3500}x more data!)")
    print(f"\n🚀 Next step: run python train.py")


if __name__ == "__main__":
    convert()