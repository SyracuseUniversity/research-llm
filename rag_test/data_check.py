import sqlite3
import re
from collections import Counter

DB_PATH = r"D:\OSPO\KG-RAG1\researchers_fixed.db"  # adjust path if needed

# Patterns for common sources
PATTERNS = {
    "arxiv": re.compile(r"arxiv\.org", re.I),
    "biorxiv": re.compile(r"biorxiv\.org", re.I),
    "eartharxiv": re.compile(r"eartharxiv\.org", re.I),
    "evoarxiv": re.compile(r"evoarxiv\.org", re.I),
    "doi.org": re.compile(r"doi\.org", re.I),
    "researchgate": re.compile(r"researchgate\.net", re.I),
    "elsevier": re.compile(r"elsevier\.com", re.I),
    "springer": re.compile(r"springer\.com", re.I),
    "wiley": re.compile(r"wiley\.com", re.I),
    "nature": re.compile(r"nature\.com", re.I),
    "other": re.compile(r".*"),  # fallback
}

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Try to find links in relevant columns (info, doi, or url)
cur.execute("""
SELECT info, doi, work_title
FROM research_info
""")

counts = Counter()
total = 0

for info, doi, title in cur.fetchall():
    combined = " ".join(filter(None, [info, doi, title]))
    total += 1
    matched = False
    for label, pattern in PATTERNS.items():
        if pattern.search(combined):
            counts[label] += 1
            matched = True
            break
    if not matched:
        counts["unknown"] += 1

conn.close()

# Display percentage breakdown
print("Source breakdown (percentage of total):\n")
for label, count in counts.most_common():
    pct = (count / total) * 100
    print(f"{label:<15}: {count:>5} ({pct:.2f}%)")
