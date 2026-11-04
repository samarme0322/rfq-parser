# RFQ Parser — Approach and Evaluation

## Approach

`parse.py` implements `parse_rfq(text)` with Python's standard library only.
It normalizes common spelling and Hinglish variants, matches product aliases
for all seven required categories, expands Indian number formats (commas, K,
lakh/hazar and crore), and extracts unit-bearing quantities. Capacity, GSM,
micron and ply specifications are excluded from order quantity selection.
For explicitly ranged amounts with a multiplier (for example `40-50k`), the
parser uses the upper bound. Film quantities prefer MT, kg or rolls; other
categories prefer pieces and infer `pcs` for bare numbers. Locations are
extracted from delivery cues or a city/country list; explicit delivery cues
also support unlisted place names. Missing fields are returned as `None`
(except unit, which defaults to `pcs`).

## Test score

`python3 score.py my_tests.jsonl`: **64/64 exact matches (100%)**. The 64
self-authored examples cover the seven categories, Hinglish, Indian number
formats, missing fields, capacity-vs-order quantity, and parser edge cases.
This is a development-set score, not an estimate of accuracy on unseen buyer
messages.

## Known failure modes

1. `BOPP bags deliver to Pnue` returns `Pnue`: explicit delivery cues accept
   unlisted place names, so the parser cannot distinguish a typo from a real
   place without fuzzy matching or external context.
2. `BOPP bags 500 bundles` may be read as 500 pcs: uncommon units such as
   bundles are not modeled.
3. `Need 40-50 BOPP bags` is ambiguous without a multiplier or unit; the parser
   may select the first number rather than infer a range's intended meaning.

With 1,000 real labelled messages, I would first define annotation guidance,
double-label a subset, and measure field-level precision/recall by category,
language and message type. I would use the errors to improve rules or compare
them against lightweight NER/classification models on held-out data.

For real buyers, I would run the parser in shadow mode beside human operators,
then offer extracted fields for buyer confirmation. I would monitor corrections
and field-level accuracy by category and language before using results
operationally.

**Tools:** Python 3 standard library (`re`, `unicodedata`, `typing`) and the
provided scorer; no third-party packages or paid APIs.
**AI disclosure:** I used the Copilot AI assistant in VS Code; I personally
reviewed the extraction rules, added regression cases, and ran the scorer.
