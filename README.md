# RFQ Parser

A lightweight, rule-based parser for turning customer packaging requests into
structured RFQ data. It handles common English and Hinglish phrasing, Indian
number formats, product aliases, and quantity-vs-capacity distinctions.

## Requirements

- Python 3.10 or newer
- No third-party packages

## Use

Import `parse_rfq` from `parse.py`:

```python
from parse import parse_rfq

result = parse_rfq("bopp bags chahiye 40 hazaar, 65 gsm, Nagpur")
print(result)
```

Output:

```python
{
    "category": "bopp_bag",
    "quantity": 40000,
    "unit": "pcs",
    "gsm": 65,
    "micron": None,
    "location": "Nagpur",
}
```

Every result contains exactly these six keys: `category`, `quantity`, `unit`,
`gsm`, `micron`, and `location`. Missing extracted values are `None`, except
`unit`, which defaults to `pcs`.

Run the built-in smoke tests with:

```sh
python3 parse.py
```

The JSONL test set is in `my_tests.jsonl`. It contains 64 examples with
expected outputs. The assignment's `score.py` is not included in this
repository; when available, run `python3 score.py my_tests.jsonl` to evaluate
the test set.

See [NOTE.md](NOTE.md) for the approach, development-set score, limitations,
evaluation strategy, and AI/tool disclosure.
