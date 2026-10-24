"""Cleaning rules and product-family keys (simsearch/data.py)."""
from simsearch.data import brand, category_text, colours, family_key, list_price, to_float, weight_grams


def test_cleaning_rules():
    assert weight_grams("200 g") == 200 and weight_grams("1.5 Kilograms") == 1500
    assert weight_grams("999999999") is None and weight_grams("some free text") is None
    assert brand(" Generic ") is None and brand("BIBA") == "biba"
    assert colours("Black|Navy Blue") == ["black", "navy blue"]
    assert category_text({"WomensKurtasKurtis": "#1"}) == "Womens Kurtas Kurtis"
    assert list_price({"sales_price": "600", "discount_percentage": "40%"}) == 1000
    assert list_price({"sales_price": "600"}) == 600 and list_price({}) is None
    assert to_float("1,299") == 1299 and to_float("nan") is None and to_float("inf") is None


def test_family_key_groups_sizes_and_sku_codes_but_not_different_products():
    a = family_key("Max Men's Striped Slim Fit T-Shirt (BSPSM1902KTS_White_M)")
    assert a == family_key("Max Men's Striped Slim Fit T-Shirt (BSPSM1902KTS_White_2XL)")
    assert family_key("Krystle Kids Socks 2-3 Years") == family_key("Krystle Kids Socks 3-4 Years")
    assert family_key("Brief Pack of 2") == family_key("Brief Pack of 3")
    assert a != family_key("Max Men's Solid Slim Fit Polo")
