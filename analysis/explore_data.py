"""First look at the data, before any modelling. Every cleaning rule in simsearch/data.py comes from a finding here.

    python analysis/explore_data.py   -> analysis/data_report.md and analysis/figures/*.png

I read the raw JSON lines as they are, answer one question per section, and print what I found. The last section
runs the cleaning functions on the same rows, to check that they do what the findings say they should.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from simsearch.data import (brand, build_catalogue, colours, family_key, list_price, load_raw,  # noqa: E402
                            name_key, to_float, weight_grams)

OUT = ROOT / "analysis"
FIG = OUT / "figures"
BRIEF_FIELDS = ["brand", "colour", "sales_price", "weight", "rating", "medium", "product_name",
                "parent___child_category__all", "discount_percentage", "other_items_customers_buy", "meta_keywords"]
report = []


def say(line=""):
    print(line)
    report.append(line)


def present(value) -> bool:
    return value not in (None, "", {}, [])


def share(n, total) -> str:
    return f"{n:,} of {total:,} ({n / total:.0%})"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=110)
    plt.close(fig)
    say(f"![{name}](figures/{name}.png)")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    n = len(raw)
    say("# Data report")
    say(f"\n{n:,} products, {len({k for p in raw for k in p})} distinct fields. Source: `data/archive.zip`.")

    say("\n## 1. Which fields can I rely on?")
    cover = {f: sum(present(p.get(f)) for p in raw) / n for f in BRIEF_FIELDS}
    # "filled" overstates weight: most values are a placeholder (section 2), so count real values only
    cover["weight"] = sum(present(p.get("weight")) and str(p["weight"]).strip() != "999999999" for p in raw) / n
    for f, c in cover.items():
        say(f"- `{f}`: {c:.0%} filled" + (" with a real value (the rest is the placeholder 999999999)"
                                          if f == "weight" else ""))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(list(cover)[::-1], [100 * c for c in cover.values()][::-1], color="#2f6fed")
    ax.set_xlabel("% of products with a value")
    ax.set_xlim(0, 100)
    save(fig, "field_coverage")

    say("\n## 2. Weight: what do the values look like?")
    weights = [str(p.get("weight", "")).strip() for p in raw]
    placeholder = sum(w == "999999999" for w in weights)
    say(f"- `999999999` appears in {share(placeholder, n)}. That is a placeholder, not a weight: read as a number it "
        "would make most products 'identical' on weight. Decision: treat it as unknown.")
    units = Counter(re.sub(r"[\d.,\s]+", " ", w).strip().lower()[:20] or "(number only)"
                    for w in weights if w and w != "999999999")
    say(f"- The real values mix units: {', '.join(f'{u!r} ({c})' for u, c in units.most_common(4))}, plus a few free-text "
        "entries. Decision: convert units to grams; anything unparseable is unknown.")

    say("\n## 3. Brand: how often is it missing, and what does missing look like?")
    brands = [str(p.get("brand") or "").strip() for p in raw]
    empty = sum(not b for b in brands)
    generic = Counter(b.lower() for b in brands if b.lower() in {"generic", "unknown", "none", "na", "n/a"})
    say(f"- Empty in {share(empty, n)}; placeholders: {dict(generic)}.")
    say("- Decision: empty and placeholder brands are *unknown*, and unknown never counts as a match. An 'Unknown' "
        "brand value would make thousands of unbranded products look alike.")
    counts = Counter(b.lower() for b in brands if b)
    say(f"- {len(counts):,} distinct brands; the 10 largest cover {sum(c for _, c in counts.most_common(10)) / n:.0%} "
        "of products, so brand is a long tail.")

    say("\n## 4. Colour: sparse and multi-valued")
    col = [str(p.get("colour") or "") for p in raw]
    options = [len([x for x in c.split("|") if x.strip()]) for c in col if c]
    say(f"- Filled in {share(len(options), n)}. Every filled value is a `|`-separated list (median "
        f"{np.median(options):.0f}, up to {max(options)} entries), so the field holds the *colour options the product "
        "is sold in*, not the colour of this listing.")
    say("- Decision: split into individual colours and compare the sets (two products sold in similar colours). It is a "
        "weak signal, and a missing colour adds nothing to a score.")

    say("\n## 5. Prices: skew, and the missing 'price' column")
    sales = np.array([v for v in (to_float(p.get("sales_price")) for p in raw) if v])
    q = np.percentile(sales, [5, 25, 50, 75, 95, 99])
    say(f"- `sales_price` parses for {share(len(sales), n)}. Percentiles 5/25/50/75/95/99: "
        + ", ".join(f"₹{v:,.0f}" for v in q) + ". The top 1% is far above the median.")
    say("- Decision: compare prices by equal-population bands, not raw values, so ₹300 vs ₹30,000 is not "
        "dominated by a few luxury items, and give partial credit to neighbouring bands so ₹499 vs ₹501 is close.")
    disc = sum(present(p.get("discount_percentage")) for p in raw)
    say(f"- The brief lists both `price` and `sales_price`, but there is no `price` field. `discount_percentage` is "
        f"filled in {share(disc, n)}, so I derive the list price as sales price / (1 - discount).")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(np.log10(sales), bins=60, color="#1f9d6b")
    ax.set_xlabel("sales price, log10 rupees")
    ax.set_ylabel("products")
    save(fig, "price_distribution")

    say("\n## 6. Ratings: where do they sit?")
    ratings = np.array([v for v in (to_float(p.get("rating")) for p in raw) if v])
    say(f"- {share(len(ratings), n)} have a rating; {np.mean(ratings == 5):.0%} are exactly 5.0 and the median is "
        f"{np.median(ratings):.1f}. Decision: half-star bands with neighbour credit, and rating as the tie-breaker.")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    vals, cnt = np.unique(np.round(ratings * 2) / 2, return_counts=True)
    ax.bar(vals, cnt, width=0.4, color="#e2b93b")
    ax.set_xlabel("rating")
    ax.set_ylabel("products")
    save(fig, "ratings")

    say("\n## 7. Categories: keys or values?")
    cats = [p.get("parent___child_category__all") or {} for p in raw]
    say(f"- Example: `{cats[0]}`. The keys say what the product is; the values are sales ranks (popularity), so "
        "I use the keys as text, split on CamelCase (`WomensKurtasKurtis` -> 'Womens Kurtas Kurtis').")
    leaf = Counter(list(c)[-1] for c in cats if c)
    say(f"- {share(sum(not c for c in cats), n)} have no category at all. {len(leaf)} distinct leaf categories; "
        "largest: " + ", ".join(f"{k} ({v:,})" for k, v in leaf.most_common(5)) + ".")
    fig, ax = plt.subplots(figsize=(7, 4))
    top = leaf.most_common(12)[::-1]
    ax.barh([k for k, _ in top], [v for _, v in top], color="#8f5be8")
    ax.set_xlabel("products")
    save(fig, "top_categories")

    say("\n## 8. Is the same product listed more than once?")
    asins = Counter(p.get("asin") for p in raw)
    names = Counter(name_key(p.get("product_name")) for p in raw)
    fams = Counter(family_key(p.get("product_name")) for p in raw)
    say(f"- {sum(c > 1 for c in asins.values()):,} ASINs appear more than once.")
    say(f"- {sum(c > 1 for c in names.values()):,} normalised names repeat, covering "
        f"{sum(c for c in names.values() if c > 1):,} listings (sizes and colours listed separately).")
    say(f"- Removing size words, age ranges, pack counts and bracketed SKU codes leaves {len(fams):,} product "
        f"families from {len(names):,} names. Example: "
        "`Max Men's Striped Slim Fit T-Shirt (BSPSM1902KTS_White_M)` and `(..._2XL)` are one family.")
    say("- Decision: show one listing per product, or a shopper sees the same T-shirt in four sizes.")

    say("\n## 9. Photos")
    photo = sum(present(p.get("medium")) for p in raw)
    say(f"- {share(photo, n)} have at least one image URL. The first `medium` URL is the product photo I embed. "
        "Some URLs fail to download, so the photo pipeline records which products really have a photo, and a product "
        "without one falls back to its other attributes.")

    say("\n## 10. Is there anything I could use as ground truth?")
    also = [str(p.get("other_items_customers_buy") or "") for p in raw]
    listed = [[x.strip() for x in a.split("|") if x.strip()] for a in also if a]
    known = set(names)
    matched = [[x for x in row if name_key(x) in known] for row in listed]
    say(f"- `other_items_customers_buy` is filled for {share(len(listed), n)}, with {np.mean([len(r) for r in listed]):.1f} "
        "names per row on average.")
    say(f"- {sum(len(m) for m in matched) / sum(len(r) for r in listed):.0%} of those names match a product in this "
        f"catalogue, and {sum(bool(m) for m in matched):,} products have at least one match. Keeping one query per "
        "product name and dropping matches to the query's own name leaves 12,193 labelled queries "
        "(`simsearch/labels.py`).")
    say("- This is real shopper behaviour, so it can serve as weak relevance labels. It is never used as a feature, "
        "or the model would be scored on its own input. Note: 'also bought' is mostly same-category "
        "alternatives, with some complements mixed in.")

    say("\n## 11. Text: which fields describe the product?")
    words = lambda s: set(re.findall(r"[a-z0-9]+", str(s or "").lower()))  # noqa: E731
    new = np.array([len(words(p.get("meta_keywords")) - words(p.get("product_name")))
                    / max(1, len(words(p.get("meta_keywords")))) for p in raw])
    lengths = [len(str(p.get("product_name", "")).split()) for p in raw]
    say(f"- `meta_keywords` is the product name plus a few seller words: in {np.mean(new == 0):.0%} of rows it adds no "
        f"new word at all, and the median row adds {np.median(new):.0%} new words. Names are {np.median(lengths):.0f} "
        "words long (median). Decision: TF-IDF on name + category text; keywords left out.")

    say("\n## 12. Do the cleaning functions do what this report says?")
    cat = build_catalogue(raw)
    checks = {
        "placeholder weights become unknown": sum(w is None for w in cat.weights_g) >= placeholder,
        "every real weight is in grams and positive": all(w > 0 for w in cat.weights_g if w is not None),
        "placeholder brands are unknown": brand("Generic") is None and brand("") is None,
        "multi-colours are split": colours("Black|White") == ["black", "white"],
        "list price is derived from the discount": list_price({"sales_price": "600", "discount_percentage": "40%"}) == 1000,
        "weights with units convert": weight_grams("1.5 Kilograms") == 1500 and weight_grams("200 g") == 200,
    }
    for check, ok in checks.items():
        say(f"- {'PASS' if ok else 'FAIL'}: {check}")
    (OUT / "data_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise SystemExit("a cleaning check failed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    main()
