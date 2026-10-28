"""What the input looks like, and what the channels are called.

The one place for the input contract (which columns each file must have) and the channel
vocabulary. Everything else in the engine uses column names directly, as pandas code does;
the contract is checked once, when a file comes in, so a missing column is reported at upload
("inventory is missing columns: units_on_hand") rather than as a KeyError somewhere deep.
"""

from dataclasses import dataclass

# --- channels -------------------------------------------------------------------------------
AMAZON = "amazon_ae"
NOON = "noon"  # parity is measured as Noon against Amazon.ae
CHANNEL_LABELS = {AMAZON: "Amazon.ae", NOON: "Noon"}

# How marketplaces get written in exports (after lower-casing and removing spaces) -> our id
CHANNEL_ALIASES = {
    "amazon_ae": AMAZON,
    "amazon.ae": AMAZON,
    "amazonae": AMAZON,
    "amazon": AMAZON,
    "noon": NOON,
    "noon.com": NOON,
}


# --- the three files ------------------------------------------------------------------------
@dataclass(frozen=True)
class FileSchema:
    """What one input file must contain: required, optional and numeric columns, and the columns that identify it
    when uploaded.
    """

    kind: str  # short key used by the upload code
    name: str  # how errors refer to it
    signature: frozenset  # enough columns to recognise the file on upload
    required: tuple  # the file is rejected without these
    optional: tuple = ()  # used when present
    numeric: tuple = ()  # coerced to numbers on load

    @property
    def columns(self) -> tuple:
        """Required then optional columns: everything the loader keeps."""
        return self.required + self.optional


HISTORY = FileSchema(
    kind="history",
    name="daily history",
    signature=frozenset({"date", "style_id", "channel", "units_sold"}),
    required=("date", "style_id", "channel", "price_aed", "sessions", "ad_spend_aed", "ad_clicks", "units_sold"),
    numeric=("price_aed", "sessions", "ad_spend_aed", "ad_clicks", "units_sold"),
)
INVENTORY = FileSchema(
    kind="inventory",
    name="inventory",
    signature=frozenset({"style_id", "channel", "size_eu", "units_on_hand"}),
    required=("style_id", "channel", "size_eu", "units_at_start", "units_on_hand", "unit_cost_aed"),
    optional=("style_name", "category"),
    numeric=("units_at_start", "units_on_hand", "unit_cost_aed"),
)
CONSTRAINTS = FileSchema(
    kind="constraints",
    name="constraints",
    signature=frozenset({"style_id", "liquidation_floor_aed"}),
    required=("style_id", "mrp_aed", "liquidation_floor_aed"),
    optional=("style_name", "category"),
    numeric=("mrp_aed", "liquidation_floor_aed"),
)
FILES = (HISTORY, INVENTORY, CONSTRAINTS)

# Other names a marketplace or ERP export plausibly uses for the same column. Matched after
# tidying (lower case, spaces and punctuation to "_"), so "Units Sold" already matches units_sold.
COLUMN_SYNONYMS = {
    "date": ["day", "report_date"],
    "style_id": ["style", "sku", "style_code"],
    "style_name": ["name", "product_name"],
    "channel": ["marketplace"],
    "price_aed": ["price", "selling_price"],
    "sessions": ["page_views", "views", "visits"],
    "ad_spend_aed": ["ad_spend", "spend"],
    "ad_clicks": ["clicks"],
    "units_sold": ["units", "orders", "qty_sold"],
    "size_eu": ["size", "eu_size"],
    "units_at_start": ["opening_stock", "units_start"],
    "units_on_hand": ["on_hand", "stock", "closing_stock"],
    "unit_cost_aed": ["unit_cost", "cost"],
    "mrp_aed": ["mrp"],
    "liquidation_floor_aed": ["floor_price_aed", "floor_price", "floor"],
}
