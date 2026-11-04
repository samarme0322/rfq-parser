"""
parse.py — RFQ Parser for Kypson Assignment
============================================
Extracts six structured fields from free-form customer packaging RFQ messages.

Approach : Rule-based NLP — regular expressions + keyword dictionaries.
Handles  : English, Hinglish, Indian number formats (hazar/lakh/K/Indian-comma),
           spelling variations, quantity-vs-capacity disambiguation.

Author   : Built for Kypson take-home assignment
Tools    : Python 3.10+ stdlib only  (re, unicodedata)
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 1.  CATEGORY DEFINITIONS
#     Order matters: more specific / longer phrases come first to avoid partial
#     matches  (e.g. "jumbo bag" must precede bare "bag").
# =============================================================================

CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    (
        "fibc",
        [
            r"\bfibc\b",
            r"jumbo\s*bag",
            r"bulk\s*bag",
            r"tonne\s*bag",
            r"big\s*bag",
            r"pp\s*jumbo",
            r"flexi\s*bag",
        ],
    ),
    (
        "bopp_bag",
        [
            r"bopp\s*laminat",
            r"bopp\s*bag",
            r"\bbopp\b",
        ],
    ),
    (
        "woven_sack",
        [
            r"ad[\s\-]*star",
            r"cement\s*(?:bag|sack)",
            r"woven\s*sack",
            r"pp\s*woven",
            r"woven\s*bag",
            r"hdpe\s*woven",
            r"hdpe\s*bag",
        ],
    ),
    (
        "corrugated_box",
        [
            r"corrugated\s*box",
            r"corrugated\s*carton",
            r"carton\s*box",
            r"plain\s*box",
            r"duplex\s*box",
            r"\d\s*ply\s*(corrugated|box|carton)",
            r"\bcorrugated\b",
        ],
    ),
    (
        "pouch",
        [
            r"stand[\s\-]*up\s*pouch",
            r"standup\s*pouch",
            r"zip(?:per)?\s*pouch",
            r"spout\s*pouch",
            r"\bpouch(?:es)?\b",
        ],
    ),
    (
        "film",
        [
            r"shrink\s*film",
            r"stretch\s*film",
            r"wrapping\s*film",
            r"cling\s*film",
            r"lldpe\s*film",
            r"pe\s*film",
            r"\bfilm\b",
        ],
    ),
    (
        "mono_carton",
        [
            r"mono[\s\-]*carton",
            r"folding\s*carton",
            r"printed\s*carton",
            r"monocarton",
        ],
    ),
]


# =============================================================================
# 2.  INDIAN CITIES GAZETTEER  (~180 cities + select countries)
# =============================================================================

INDIAN_CITIES = {
    # Metros and tier-1 cities
    "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "kolkata",
    "pune", "ahmedabad", "surat", "jaipur", "lucknow", "kanpur", "nagpur", "indore",
    "thane", "bhopal", "visakhapatnam", "vizag", "patna", "vadodara", "baroda",
    "ludhiana", "agra", "nashik", "faridabad", "meerut", "rajkot", "kalyan",
    "varanasi", "srinagar", "aurangabad", "dhanbad", "amritsar", "navi mumbai",
    "allahabad", "prayagraj", "howrah", "ranchi", "coimbatore", "jabalpur", "gwalior",
    "vijayawada", "jodhpur", "madurai", "raipur", "kota", "guwahati", "chandigarh",
    "solapur", "hubli", "dharwad", "bareilly", "moradabad", "mysore", "mysuru",
    "tiruchirappalli", "tiruchy", "tiruppur", "tirupati", "salem", "aligarh",
    "bhubaneswar", "jalandhar", "noida", "gurugram", "gurgaon", "haridwar",
    "dehradun", "shimla", "jammu", "siliguri", "durgapur", "asansol", "jamshedpur",
    "bokaro", "bhilai", "cuttack", "firozabad", "kochi", "ernakulam",
    "thiruvananthapuram", "kozhikode", "thrissur", "kollam", "mangalore", "mangaluru",
    "belgaum", "belagavi", "gulbarga", "kalaburagi", "davangere", "bellary", "ballari",
    "bijapur", "vijayapura", "tumkur", "tumakuru", "shimoga", "shivamogga", "udupi",
    "hospet", "raichur", "nellore", "kakinada", "rajahmundry", "guntur", "kadapa",
    "kurnool", "anantapur", "eluru", "karimnagar", "nizamabad", "khammam", "warangal",
    "nanded", "amravati", "kolhapur", "sangli", "satara", "latur", "jalgaon",
    "dhule", "ahmednagar", "akola", "buldhana", "yavatmal", "wardha", "chandrapur",
    "rohtak", "hisar", "karnal", "panipat", "sonipat", "ambala", "yamunanagar",
    "bathinda", "patiala", "bikaner", "ajmer", "udaipur", "alwar", "bhilwara",
    "sikar", "ghaziabad", "muzaffarnagar", "saharanpur", "mathura", "gorakhpur",
    "gaya", "muzaffarpur", "bhagalpur", "purnia", "darbhanga", "begusarai",
    "ujjain", "ratlam", "dewas", "satna", "rewa", "chhindwara", "singrauli",
    "bilaspur", "korba", "durg", "pimpri", "vasai",
    # City abbreviations used in trade
    "blr", "hyd", "chn", "mum", "del", "kol", "pun",
    # Countries / international trade hubs
    "india", "usa", "uae", "dubai", "singapore", "uk", "germany", "australia",
    "bangladesh", "nepal",
    "new delhi", "chhatrapati sambhajinagar",
}

_NOT_LOCATION = {
    "pcs", "kg", "mt", "rolls", "roll", "bags", "bag", "box", "carton",
    "film", "pouch", "nos", "set", "unit", "pieces", "gsm", "micron",
    "need", "want", "chahiye", "bhejo",
}


# =============================================================================
# 3.  UNIT TOKEN SETS
# =============================================================================

PCS_TOKENS  = {
    "pcs", "piece", "pieces", "nos", "no", "unit", "units",
    "number", "numbers", "set", "sets", "qty",
}
KG_TOKENS   = {"kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"}
MT_TOKENS   = {
    "mt", "mts", "ton", "tons", "tonne", "tonnes",
    "metricton", "metrictons", "metrictonne", "metrictonnes",
}
ROLL_TOKENS = {"roll", "rolls"}


def _classify_unit(token: str) -> Optional[str]:
    t = token.lower().strip().replace(".", "").replace(" ", "")
    if t in PCS_TOKENS:   return "pcs"
    if t in KG_TOKENS:    return "kg"
    if t in MT_TOKENS:    return "mt"
    if t in ROLL_TOKENS:  return "rolls"
    return None


# =============================================================================
# 4.  NORMALIZATION
# =============================================================================

_SPELL_MAP = [
    (r"\bhazaar\b|\bhajar\b|\bhajaar\b|\bhazzar\b|\bhzaar\b", "hazar"),
    (r"\blacs\b|\blakhs\b|\blaakh\b", "lakh"),
    (r"\bstrech\b",  "stretch"),
    (r"\blaminat\b", "laminated"),
    (r"\bmic\b",     "micron"),
    (r"\bmu\b",      "micron"),
    (r"\bpolybag\b|\bpoly bag\b", "pe film"),
]


def normalize(text: str) -> str:
    """Lowercase, unicode-normalize, and apply Hinglish/spelling fixes."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    for pattern, repl in _SPELL_MAP:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


# =============================================================================
# 5.  INDIAN NUMBER EXPANSION
# =============================================================================

_WORD_MULT_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(hazar|hazaar|hajar|hzaar|lakh|lac\b|crore[s]?|cr\b)",
    re.IGNORECASE,
)
_RANGE_MULT_RE = re.compile(
    r"(?<!\w)\d[\d,]*(?:\.\d+)?\s*(?:-|to)\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*(hazar|hazaar|hajar|hzaar|lakh|lac\b|"
    r"crore[s]?|cr\b|k)\b",
    re.IGNORECASE,
)
_K_SUFFIX_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)k(?![a-z])", re.IGNORECASE)
_K_WORD_RE   = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s+k\b(?!\s*[gmi])", re.IGNORECASE)
_COMMA_RE    = re.compile(r"\d+(?:,\d{2,3})+")


def _expand_multipliers(text: str) -> str:
    """
    Replace Indian multiplier expressions with plain integers in-place.

    Examples
    --------
    "40 hazar"   -> "40000"
    "1.5 lakh"   -> "150000"
    "15K pcs"    -> "15000 pcs"
    "2 crore"    -> "20000000"
    "1,50,000"   -> "150000"
    "40,000"     -> "40000"
    """
    def _word_replace(m: re.Match) -> str:
        num  = float(m.group(1).replace(",", ""))
        word = m.group(2).lower()
        if   re.match(r"haz|haj",  word): factor = 1_000
        elif re.match(r"lakh|lac", word): factor = 100_000
        elif re.match(r"cr",       word): factor = 10_000_000
        else:                             factor = 1
        result = num * factor
        return str(int(result)) if result == int(result) else str(result)

    def _k_replace(m: re.Match) -> str:
        num = float(m.group(1).replace(",", ""))
        return str(int(num * 1_000))

    def _range_replace(m: re.Match) -> str:
        num = float(m.group(1).replace(",", ""))
        word = m.group(2).lower()
        if word == "k":
            factor = 1_000
        elif re.match(r"haz|haj", word):
            factor = 1_000
        elif re.match(r"lakh|lac", word):
            factor = 100_000
        else:
            factor = 10_000_000
        return str(int(num * factor))

    text = _RANGE_MULT_RE.sub(_range_replace, text)
    text = _WORD_MULT_RE.sub(_word_replace, text)
    text = _K_SUFFIX_RE.sub(_k_replace,    text)
    text = _K_WORD_RE.sub(_k_replace,      text)
    text = _COMMA_RE.sub(lambda m: m.group(0).replace(",", ""), text)
    return text


# =============================================================================
# 6.  CATEGORY EXTRACTION
# =============================================================================

def extract_category(text: str) -> Optional[str]:
    """Identify the packaging product category from normalized text."""
    for category, patterns in CATEGORY_PATTERNS:
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                return category
    return None


# =============================================================================
# 7.  QUANTITY + UNIT EXTRACTION
# =============================================================================

_NUM_UNIT_RE = re.compile(
    r"(?<!\w)"
    r"(\d+(?:\.\d+)?)"
    r"\s*"
    r"(pcs|pieces?|nos?\.?|units?|numbers?|sets?|qty|"
    r"kgs?|kilograms?|kilos?|"
    r"m\.?t\.?s?|metric\s+ton[ne]?s?|tonne?s?|"
    r"rolls?)?"
    r"(?!\w)",
    re.IGNORECASE,
)
_CAPACITY_RE = re.compile(r"\b(capacity|each|per\s*bag|/\s*bag)\b", re.IGNORECASE)
_GSM_SFXRE   = re.compile(r"\bgsm\b|g/m2?",   re.IGNORECASE)
_MIC_SFXRE   = re.compile(r"\bmicron[s]?\b",   re.IGNORECASE)


def extract_quantity_unit(
    text: str, category: Optional[str]
) -> Tuple[Optional[int], str]:
    """
    Extract the ORDER quantity and its measurement unit.

    Non-film: pcs  >  bare number (infer pcs)  >  kg  >  mt
    Film    : mt   >  kg  >  rolls  >  pcs  >  bare (infer kg)

    Capacity annotations ("1000 kg capacity") are excluded.
    GSM / micron suffix numbers are excluded.
    """
    expanded = _expand_multipliers(text)
    candidates: List[Dict[str, Any]] = []

    for m in _NUM_UNIT_RE.finditer(expanded):
        raw_num  = m.group(1)
        unit_tok = m.group(2) or ""

        try:
            value = float(raw_num)
        except ValueError:
            continue
        if value <= 0:
            continue

        unit  = _classify_unit(unit_tok) if unit_tok else None
        after = expanded[m.end(): m.end() + 40]

        is_capacity = bool(_CAPACITY_RE.search(after))

        # Only mark as GSM/micron property if the suffix appears IMMEDIATELY
        # after this number (within ~6 chars), with no comma intervening.
        # This avoids falsely flagging "40000" because ", 65 gsm" appears later.
        immediate = after[:6]
        is_gsm    = bool(_GSM_SFXRE.search(immediate)) and "," not in immediate
        is_micron = bool(_MIC_SFXRE.search(immediate)) and "," not in immediate

        # Skip "3 ply", "5 ply" etc. — these are board specs, not order qty
        is_ply = bool(re.match(r"\s*ply\b", after, re.IGNORECASE))

        if is_gsm or is_micron or is_ply:
            continue

        candidates.append({"value": value, "unit": unit, "is_capacity": is_capacity})

    valid = [c for c in candidates if not c["is_capacity"]] or candidates

    if not valid:
        return None, ("kg" if category == "film" else "pcs")

    pcs_c  = [c for c in valid if c["unit"] == "pcs"]
    kg_c   = [c for c in valid if c["unit"] == "kg"]
    mt_c   = [c for c in valid if c["unit"] == "mt"]
    roll_c = [c for c in valid if c["unit"] == "rolls"]
    bare_c = [c for c in valid if c["unit"] is None]

    if category == "film":
        if mt_c:   return _to_int(mt_c[0]["value"]),   "mt"
        if kg_c:   return _to_int(kg_c[0]["value"]),   "kg"
        if roll_c: return _to_int(roll_c[0]["value"]), "rolls"
        if pcs_c:  return _to_int(pcs_c[0]["value"]),  "pcs"
        if bare_c: return _to_int(bare_c[0]["value"]), "kg"
        return None, "kg"

    if pcs_c:
        best = max(pcs_c, key=lambda x: x["value"])
        return _to_int(best["value"]), "pcs"
    if roll_c:  return _to_int(roll_c[0]["value"]),  "rolls"
    if bare_c:  return _to_int(bare_c[0]["value"]),  "pcs"
    if kg_c:    return _to_int(kg_c[0]["value"]),    "kg"
    if mt_c:    return _to_int(mt_c[0]["value"]),    "mt"

    return None, "pcs"


def _to_int(val: float) -> int:
    return int(round(val))


# =============================================================================
# 8.  GSM & MICRON EXTRACTION
# =============================================================================

_GSM_RE    = re.compile(r"(\d+(?:\.\d+)?)\s*(?:gsm|g/m2?)\b",        re.IGNORECASE)
_MICRON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:micron[s]?|mic\b)",   re.IGNORECASE)
_GSM_PREFIX_RE = re.compile(r"\bgsm\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_MICRON_PREFIX_RE = re.compile(
    r"\bmicron[s]?\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE
)


def extract_gsm(text: str) -> Optional[int]:
    """Extract material GSM (grams per square metre)."""
    m = _GSM_RE.search(text)
    if not m:
        m = _GSM_PREFIX_RE.search(text)
    return int(float(m.group(1))) if m else None


def extract_micron(text: str) -> Optional[int]:
    """Extract material thickness in microns."""
    m = _MICRON_RE.search(text)
    if not m:
        m = _MICRON_PREFIX_RE.search(text)
    return int(float(m.group(1))) if m else None


# =============================================================================
# 9.  LOCATION EXTRACTION
# =============================================================================

_LOC_MARKERS = [
    re.compile(
        r"deliver(?:y)?\s+(?:to|at|in)\s+([A-Za-z][A-Za-z\s\-']{2,79}?)(?:\s*[,.\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:send|ship|dispatch)\s+to\s+([A-Za-z][A-Za-z\s\-']{2,79}?)(?:\s*[,.\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:location|loc|city|destination|dest)\s*[:\-]\s*([A-Za-z][A-Za-z\s\-']{2,79}?)(?:\s*[,.\n]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bbhejo\s+([A-Za-z][A-Za-z\s\-']{2,79}?)(?:\s+mein|\s+me|\s*[,.\n]|$)",
        re.IGNORECASE,
    ),
]
_HINGLISH_FILLERS = re.compile(r"\s*(mein|me|ko|ke|liye|tak)\s*$", re.IGNORECASE)
_LOCATION_CONTEXT = re.compile(
    r"\b(?:our|warehouse|factory|office|near|tomorrow|today|please|"
    r"quantity|qty|order|required|needed|for)\b.*$",
    re.IGNORECASE,
)


def extract_location(original_text: str) -> Optional[str]:
    """
    Extract delivery city or country using two-tier approach.

    Tier 1 — Explicit delivery markers (highest confidence).
    Tier 2 — Gazetteer scan; longest match wins.
    """
    for pat in _LOC_MARKERS:
        m = pat.search(original_text)
        if m:
            candidate = _HINGLISH_FILLERS.sub("", m.group(1)).strip()
            candidate = _LOCATION_CONTEXT.sub("", candidate).strip(" ,.-")
            candidate_lower = candidate.lower()
            known_match = next(
                (
                    city
                    for city in sorted(INDIAN_CITIES, key=len, reverse=True)
                    if re.search(r"\b" + re.escape(city) + r"\b", candidate_lower)
                ),
                None,
            )
            if known_match:
                return known_match.title()
            if candidate_lower not in _NOT_LOCATION and len(candidate) >= 3:
                return candidate.title()

    text_lower = original_text.lower()
    for city in sorted(INDIAN_CITIES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", text_lower):
            return city.title()

    return None


# =============================================================================
# 10. MAIN PUBLIC API
# =============================================================================

def parse_rfq(text: str) -> dict:
    """
    Parse a customer RFQ message into structured data.

    Parameters
    ----------
    text : str
        Raw customer message. Supports English + Hinglish, Indian number
        formats (hazar / lakh / K / Indian-comma), common spelling mistakes.

    Returns
    -------
    dict with exactly six keys:
        category (str | None)  -- product category slug
        quantity (int | None)  -- order quantity
        unit     (str)         -- 'pcs', 'kg', 'mt', or 'rolls'
        gsm      (int | None)  -- material weight in g/m2
        micron   (int | None)  -- material thickness in microns
        location (str | None)  -- delivery city / country
    """
    norm = normalize(text)

    category       = extract_category(norm)
    quantity, unit = extract_quantity_unit(norm, category)
    gsm            = extract_gsm(norm)
    micron         = extract_micron(norm)
    location       = extract_location(text)

    return {
        "category": category,
        "quantity": quantity,
        "unit":     unit,
        "gsm":      gsm,
        "micron":   micron,
        "location": location,
    }


# =============================================================================
# 11. SMOKE TESTS  (python parse.py)
# =============================================================================

if __name__ == "__main__":
    _SMOKE = [
        (
            "bopp bags chahiye 40 hazaar, 65 gsm, Nagpur",
            {"category": "bopp_bag", "quantity": 40000, "unit": "pcs",
             "gsm": 65, "micron": None, "location": "Nagpur"},
        ),
        (
            "jumbo bag 1000 kg capacity, 500 pcs, Vadodara",
            {"category": "fibc", "quantity": 500, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Vadodara"},
        ),
        (
            "Need 5000 corrugated boxes",
            {"category": "corrugated_box", "quantity": 5000, "unit": "pcs",
             "gsm": None, "micron": None, "location": None},
        ),
        (
            "shrink film 2 MT Delhi",
            {"category": "film", "quantity": 2, "unit": "mt",
             "gsm": None, "micron": None, "location": "Delhi"},
        ),
        (
            "20 hazaar cement bags bhejo Mumbai",
            {"category": "woven_sack", "quantity": 20000, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Mumbai"},
        ),
        (
            "500 pouches, 25 micron, Pune deliver karna hai",
            {"category": "pouch", "quantity": 500, "unit": "pcs",
             "gsm": None, "micron": 25, "location": "Pune"},
        ),
        (
            "mono carton 50000 pcs, 300 gsm, Noida",
            {"category": "mono_carton", "quantity": 50000, "unit": "pcs",
             "gsm": 300, "micron": None, "location": "Noida"},
        ),
        (
            "1 lakh BOPP laminated bags, Delhi",
            {"category": "bopp_bag", "quantity": 100000, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Delhi"},
        ),
        (
            "stretch film 500 kg Mumbai",
            {"category": "film", "quantity": 500, "unit": "kg",
             "gsm": None, "micron": None, "location": "Mumbai"},
        ),
        (
            "25 kg bags, 10000 pcs woven sack, Surat",
            {"category": "woven_sack", "quantity": 10000, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Surat"},
        ),
        (
            "15K BOPP bags, 80 gsm, delivery to Pune",
            {"category": "bopp_bag", "quantity": 15000, "unit": "pcs",
             "gsm": 80, "micron": None, "location": "Pune"},
        ),
        (
            "1,50,000 bopp bags required Ahmedabad",
            {"category": "bopp_bag", "quantity": 150000, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Ahmedabad"},
        ),
        (
            "AD Star bags 25000, 100 gsm, Jaipur",
            {"category": "woven_sack", "quantity": 25000, "unit": "pcs",
             "gsm": 100, "micron": None, "location": "Jaipur"},
        ),
        (
            "zipper pouch 5000 pcs, 50 micron, Mumbai",
            {"category": "pouch", "quantity": 5000, "unit": "pcs",
             "gsm": None, "micron": 50, "location": "Mumbai"},
        ),
        (
            "folding carton 1 lakh nos, Pune",
            {"category": "mono_carton", "quantity": 100000, "unit": "pcs",
             "gsm": None, "micron": None, "location": "Pune"},
        ),
    ]

    passed = 0
    for text, expected in _SMOKE:
        result = parse_rfq(text)
        ok     = result == expected
        icon   = "PASS" if ok else "FAIL"
        print(f"[{icon}]  {text!r}")
        if not ok:
            for k, exp_v in expected.items():
                got_v = result.get(k)
                if got_v != exp_v:
                    print(f"       {k}: got={got_v!r}  expected={exp_v!r}")
        else:
            passed += 1

    print(f"\n{passed}/{len(_SMOKE)} smoke tests passed.")
