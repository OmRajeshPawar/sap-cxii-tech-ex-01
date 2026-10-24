"""Small synthetic catalogues for unit tests: a few hand-written products, so each test reads as a story."""
from simsearch.data import build_catalogue
from simsearch.engine import PRIOR_WEIGHTS, SimilarityEngine


def product(uid, name, brand_="", colour="", price="100", rating="4.0", asin=None, also="", category="ClothingAccessories"):
    return {"uniq_id": uid, "asin": asin or uid, "product_name": name, "brand": brand_, "colour": colour,
            "sales_price": price, "rating": rating, "weight": "999999999",
            "parent___child_category__all": {category: "#1"} if category else None,
            "other_items_customers_buy": also}


def engine_for(products):
    return SimilarityEngine(build_catalogue(products), weights=dict(PRIOR_WEIGHTS))
