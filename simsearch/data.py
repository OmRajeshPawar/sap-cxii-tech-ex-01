"""Loading and cleaning. Each rule comes from a finding in analysis/data_report.md and is explained in DESIGN.md (D1-D5)."""
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DATA_ZIP = Path(__file__).resolve().parent.parent / "data" / "archive.zip"

WEIGHT_PLACEHOLDER = "999999999"  # 79% of rows: means "unknown", not a weight
UNKNOWN_BRANDS = {"", "generic", "unknown", "none", "na", "n/a"}
GRAMS_PER_UNIT = {"g": 1, "gram": 1, "grams": 1, "kg": 1000, "kilogram": 1000, "kilograms": 1000}


def load_raw(path: Path = DATA_ZIP) -> List[dict]:
    """Read the line-delimited JSON straight out of the provided zip."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".ldjson"))
        with archive.open(name) as f:
            return [json.loads(line) for line in f if line.strip()]


def to_float(value) -> Optional[float]:
    """'1,299' -> 1299.0; text, 'nan' or 'inf' -> None."""
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def weight_grams(value) -> Optional[float]:
    """'200 g' -> 200, '1.2 Kilograms' -> 1200; the placeholder or free text -> None."""
    text = str(value or "").strip()
    match = re.fullmatch(r"([\d.,]+)\s*([A-Za-z]+)", text)
    if text == WEIGHT_PLACEHOLDER or not match:
        return None
    unit, number = GRAMS_PER_UNIT.get(match.group(2).lower()), to_float(match.group(1))
    return number * unit if unit and number is not None else None


def brand(value) -> Optional[str]:
    b = str(value or "").strip().lower()
    return None if b in UNKNOWN_BRANDS else b


def colours(value) -> List[str]:
    """'Black|Navy Blue' -> ['black', 'navy blue']."""
    return [c.strip().lower() for c in str(value or "").split("|") if c.strip()]


def category_text(categories) -> str:
    """{'WomensKurtasKurtis': '#1793'} -> 'Womens Kurtas Kurtis' (keys only; values are sales ranks)."""
    if not isinstance(categories, dict):
        return ""
    return " ".join(re.sub(r"(?<=[a-z])(?=[A-Z])|_", " ", key) for key in categories)


def name_key(name) -> str:
    """Normalised name: identifies size/colour variants and links 'customers also buy' entries."""
    return re.sub(r"[^a-z0-9]+", " ", str(name or "").lower()).strip()


# Size words and age ranges that separate variants of one product ("... (White_M)", "... 3-4 Years", "Pack of 2")
SIZE_WORDS = re.compile(r"\b(?:xxs|xs|s|m|l|xl|xxl|xxxl|[2-6]xl|free ?size|small|medium|large|x-large"
                        r"|\d+(?:-\d+)?\s*(?:years?|yrs?|y|months?|mths?|m)|size\s*\w+|pack of \d+|set of \d+"
                        r"|\d{2,3}(?:cm)?)\b")


def family_key(name) -> str:
    """Product family: the name without bracketed SKU/size codes, size words or pack counts (D13).

    'Max Men's Striped T-Shirt (BSPSM1902KTS_White_M)' and '... (BSPSM1902KTS_White_2XL)' share a family,
    so only one of them is shown. Coarser than name_key on purpose: it also drives the family-aware split.
    """
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", str(name or "").lower())
    return name_key(SIZE_WORDS.sub(" ", text))


# Who a product is for, and what kind of product it is, as written in product names (D28). Used only where a
# product has no category, so the product-type rule still has something to compare.
WEARER_WORDS = {"men": "men", "mens": "men", "man": "men", "male": "men", "gents": "men",
                "women": "women", "womens": "women", "woman": "women", "ladies": "women", "female": "women",
                "boys": "boys", "boy": "boys", "girls": "girls", "girl": "girls",
                "baby": "baby", "infant": "baby", "infants": "baby", "newborn": "baby"}
TYPE_WORDS = {"tshirt": "tshirt", "tshirts": "tshirt", "tee": "tshirt", "tees": "tshirt", "polo": "tshirt",
              "jeans": "jeans", "jean": "jeans", "trousers": "trousers", "trouser": "trousers", "pants": "trousers",
              "joggers": "trousers", "chinos": "trousers", "shorts": "shorts", "kurta": "kurta", "kurti": "kurta",
              "kurtas": "kurta", "kurtis": "kurta", "saree": "saree", "sari": "saree", "lehenga": "lehenga",
              "dress": "dress", "dresses": "dress", "gown": "dress", "frock": "dress", "top": "top", "tops": "top",
              "shirt": "shirt", "shirts": "shirt", "socks": "socks", "cap": "cap", "hat": "cap", "shoes": "shoes",
              "sneakers": "shoes", "sandals": "shoes", "slippers": "shoes", "bra": "innerwear", "briefs": "innerwear",
              "brief": "innerwear", "boxers": "innerwear", "panty": "innerwear", "panties": "innerwear",
              "vest": "innerwear", "jacket": "jacket", "hoodie": "jacket", "sweatshirt": "jacket", "blazer": "jacket",
              "skirt": "skirt", "leggings": "leggings", "legging": "leggings", "dupatta": "dupatta", "scarf": "scarf",
              "stole": "scarf", "wallet": "wallet", "belt": "belt", "watch": "watch", "bag": "bag", "handbag": "bag",
              "fabric": "fabric", "pyjama": "nightwear", "nightsuit": "nightwear", "nightdress": "nightwear",
              "dhoti": "dhoti", "lungi": "dhoti"}


def name_tags(name, vocab) -> set:
    """The wearer or product-type groups a name mentions: "Men's Slim Fit T-Shirt" -> {"men"} and {"tshirt"}."""
    words = re.findall(r"[a-z]+", re.sub(r"t[\s-]?shirts?", " tshirt ", str(name or "").lower()))
    return {vocab[w] for w in words if w in vocab}


@dataclass
class Catalogue:
    """Column-oriented, cleaned view of the products."""
    ids: List[str]
    asins: List[str]
    names: List[str]               # display names
    name_keys: List[str]           # variant key
    family_keys: List[str]         # coarser key: one product across sizes, packs and SKU codes (D13)
    text: List[str]                # name + category words
    brands: List[Optional[str]]
    colours: List[List[str]]
    prices: List[Optional[float]]        # sales price (what the shopper pays)
    list_prices: List[Optional[float]]   # original price = sales / (1 - discount) (D5)
    ratings: List[Optional[float]]
    weights_g: List[Optional[float]]
    categories: List[Optional[str]]   # leaf category, for evaluation only
    also_bought: List[List[str]]      # name keys from 'other_items_customers_buy', for labels only
    brand_names: List[str]            # display only (demo UI)
    image_urls: List[str]             # display only (demo UI)

    def __len__(self) -> int:
        return len(self.ids)


def list_price(product) -> Optional[float]:
    """Original price before discount. No discount listed -> the sales price is the list price."""
    sales = to_float(product.get("sales_price"))
    discount = to_float(str(product.get("discount_percentage") or "").rstrip("%"))
    if sales is None:
        return None
    return sales / (1 - discount / 100) if discount and 0 < discount < 100 else sales


def build_catalogue(raw: List[dict]) -> Catalogue:
    def leaf(cats):
        return list(cats)[-1] if isinstance(cats, dict) and cats else None

    return Catalogue(
        ids=[p["uniq_id"] for p in raw],
        asins=[p.get("asin") or p["uniq_id"] for p in raw],
        names=[p.get("product_name", "") for p in raw],
        # fall back to the id so products without a name are never merged as "variants"
        name_keys=[name_key(p.get("product_name")) or p["uniq_id"] for p in raw],
        family_keys=[family_key(p.get("product_name")) or name_key(p.get("product_name")) or p["uniq_id"]
                     for p in raw],
        text=[f"{p.get('product_name', '')} {category_text(p.get('parent___child_category__all'))}" for p in raw],
        brands=[brand(p.get("brand")) for p in raw],
        colours=[colours(p.get("colour")) for p in raw],
        prices=[to_float(p.get("sales_price")) for p in raw],
        list_prices=[list_price(p) for p in raw],
        ratings=[r if (r := to_float(p.get("rating"))) else None for p in raw],
        weights_g=[weight_grams(p.get("weight")) for p in raw],
        categories=[leaf(p.get("parent___child_category__all")) for p in raw],
        also_bought=[[k for s in str(p.get("other_items_customers_buy") or "").split("|") if (k := name_key(s))]
                     for p in raw],
        brand_names=[str(p.get("brand") or "").strip() for p in raw],
        image_urls=[str(p.get("medium") or "").split("|")[0].strip() for p in raw],
    )


def load_catalogue(path: Path = DATA_ZIP) -> Catalogue:
    return build_catalogue(load_raw(path))
