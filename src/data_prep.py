"""
Data preparation pipeline for the retail demand forecasting project.

Turns the raw Favorita Kaggle files (see data/README.md for how to get
them) into one clean, analysis-ready table: daily unit sales per
store x product family, with promotion, oil-price, and holiday context
attached.

Why aggregate to store x family x day, instead of every individual item?
------------------------------------------------------------------------
The raw data is 125M+ rows across 54 stores and 4,100 items (2013-2017).
At the individual store-item level, most day-to-day series are sparse
and intermittent (an item sells zero units on many days), which mostly
tests a different modeling problem (intermittent-demand forecasting)
than the one this project is demonstrating. Aggregating to store x
family x day keeps the real-world heterogeneity that makes forecasting
worthwhile -- different stores, different product categories, promotions,
holidays, oil-price shocks -- while producing dense, well-populated daily
series that three real forecasting approaches can be fairly compared on.

Scope: top 10 stores and top 8 product families by total historical
unit sales (see notebooks/01_data_exploration.ipynb for how these were
chosen). Together they cover a large share of total volume while
keeping the raw-data pass fast enough to run on a laptop.
"""
from __future__ import annotations

import pandas as pd

TOP_STORES = [44, 45, 47, 3, 49, 46, 48, 51, 8, 50]
TOP_FAMILIES = [
    "GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING",
    "DAIRY", "BREAD/BAKERY", "POULTRY", "MEATS",
]


def load_reference_tables(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Load the small reference tables (everything except train.csv)."""
    items = pd.read_csv(f"{raw_dir}/items.csv")[["item_nbr", "family"]]
    stores = pd.read_csv(f"{raw_dir}/stores.csv")

    oil = pd.read_csv(f"{raw_dir}/oil.csv")
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})

    holidays = pd.read_csv(f"{raw_dir}/holidays_events.csv")
    # Only count a day as a demand-relevant holiday if it's a real
    # national/regional holiday that wasn't moved ("transferred") to
    # another date, and isn't a "Work Day" (a payback day, not a holiday).
    holiday_dates = set(
        holidays.loc[
            (holidays["transferred"] == False) & (holidays["type"] != "Work Day"),
            "date",
        ]
    )
    return {"items": items, "stores": stores, "oil": oil, "holiday_dates": holiday_dates}


def build_daily_panel(
    raw_dir: str,
    stores: list[int] = TOP_STORES,
    families: list[str] = TOP_FAMILIES,
    chunksize: int = 3_000_000,
) -> pd.DataFrame:
    """
    Stream train.csv in chunks (it's ~5GB uncompressed, too large to load
    at once on a typical laptop), filter to the chosen stores/families as
    we go, and aggregate to daily unit sales per store x family.
    """
    ref = load_reference_tables(raw_dir)
    items, store_meta, oil = ref["items"], ref["stores"], ref["oil"]
    keep_items = set(items.loc[items["family"].isin(families), "item_nbr"])

    reader = pd.read_csv(
        f"{raw_dir}/train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype={"store_nbr": "int16", "item_nbr": "int32"},
        chunksize=chunksize,
    )

    partial_aggs = []
    for chunk in reader:
        chunk = chunk[chunk["store_nbr"].isin(stores) & chunk["item_nbr"].isin(keep_items)]
        if chunk.empty:
            continue
        chunk = chunk.merge(items, on="item_nbr", how="left")
        # A handful of rows are returns (negative unit_sales). For a
        # demand *signal* we clip them to zero rather than let a few
        # returns distort a store-family-day total.
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)
        chunk["onpromotion"] = chunk["onpromotion"].fillna(False).astype(bool)
        agg = (
            chunk.groupby(["date", "store_nbr", "family"])
            .agg(unit_sales=("unit_sales", "sum"), n_items_on_promo=("onpromotion", "sum"))
            .reset_index()
        )
        partial_aggs.append(agg)

    daily = pd.concat(partial_aggs, ignore_index=True)
    # A second groupby combines partial aggregates for the same
    # date/store/family that landed in different chunks.
    daily = (
        daily.groupby(["date", "store_nbr", "family"])
        .agg(unit_sales=("unit_sales", "sum"), n_items_on_promo=("n_items_on_promo", "sum"))
        .reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    daily = daily.merge(
        store_meta[["store_nbr", "city", "state", "type", "cluster"]], on="store_nbr", how="left"
    )
    daily = daily.merge(oil[["date", "oil_price"]], on="date", how="left")
    daily["oil_price"] = daily["oil_price"].ffill()
    daily["is_holiday"] = daily["date"].dt.strftime("%Y-%m-%d").isin(ref["holiday_dates"])

    return daily.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)


def save_processed(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    panel = build_daily_panel(raw_dir="data/raw")
    save_processed(panel, "data/processed/daily_sales_by_store_family.csv")
    print(f"saved {len(panel):,} rows to data/processed/daily_sales_by_store_family.csv")
