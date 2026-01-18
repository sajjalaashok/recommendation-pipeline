import os
import json
import pandas as pd
from datetime import datetime
from dateutil.parser import parse as dtparse
from fpdf import FPDF

# ------------------------------------------------------------
# Optional Great Expectations (guarded)
# ------------------------------------------------------------
GE_ENABLED = False
GE_CTX = None
Validator = None
PandasExecutionEngine = None

try:
    # Avoid importing pyspark on Windows; if present, we disable GE
    import importlib
    if importlib.util.find_spec("pyspark") is None:
        from great_expectations.validator.validator import Validator
        from great_expectations.execution_engine.pandas_execution_engine import PandasExecutionEngine
        GE_ENABLED = True
    else:
        GE_ENABLED = False
except Exception:
    GE_ENABLED = False

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
PATH = "raw_zone/"
FILES = {
    "product_catalog": PATH + "recomart_product_catalog.csv",
    "raw_customers": PATH + "recomart_raw_customers.csv",
    "raw_products": PATH + "recomart_raw_products.csv",
    "external_metadata": PATH + "external_metadata.csv"
}

# Transaction files are now partitioned
TXN_PATH = PATH + "transactions/"

# Ensure structure exists
os.makedirs(TXN_PATH, exist_ok=True)
os.makedirs(os.path.join(PATH, "clickstream"), exist_ok=True)
os.makedirs(os.path.join(PATH, "landing"), exist_ok=True)

EXPECTED_SCHEMAS = {
    "product_catalog": {
        "columns": [
            "product_id","product_name","super_category","category","brand",
            "base_price","discount_percent","monthly_sales_volume","avg_rating",
            "return_rate_percent","profit_margin_percent","is_perishable",
            "shelf_life_days","launch_date"
        ],
        "types": {
            "product_id": "str","product_name": "str","super_category": "str",
            "category": "str","brand": "str","base_price": "numeric",
            "discount_percent": "numeric","monthly_sales_volume": "numeric",
            "avg_rating": "numeric","return_rate_percent": "numeric",
            "profit_margin_percent": "numeric","is_perishable": "str",
            "shelf_life_days": "numeric","launch_date": "date",
        },
    },
    "raw_customers": {
        "columns": ["customer_id","age","gender"],
        "types": {"customer_id": "str","age": "numeric","gender": "str"},
    },
    "raw_products": {
        "columns": ["product_id","product_name","category"],
        "types": {"product_id": "str","product_name": "str","category": "str"},
    },
    "transactions": {
        "columns": ["txn_id","txn_date","customer_id","product_id","quantity"],
        "types": {
            "txn_id": "str","txn_date": "date","customer_id": "str",
            "product_id": "str","quantity": "numeric",
        },
    },
    "external_metadata": {
        "columns": ["product_id", "sentiment_score", "popularity_index", "last_updated"],
        "types": {
            "product_id": "str",
            "sentiment_score": "numeric",
            "popularity_index": "numeric",
            "last_updated": "date"
        },
    }
}

RULES = {
    "product_catalog": {
        "base_price": {"min": 0, "max": 10000},
        "discount_percent": {"min": 0, "max": 100},
        "avg_rating": {"min": 1, "max": 5},
        "return_rate_percent": {"min": 0, "max": 100},
        "profit_margin_percent": {"min": 0, "max": 100},
        "shelf_life_days": {"min": 0, "max": 3650},
        "is_perishable": {"allowed": ["Yes","No"]},
        "launch_date": {"format": "%Y-%m-%d"},
    },
    "raw_customers": {
        "age": {"min": 18, "max": 100},
        "gender": {"allowed": ["M","F","O"]},
    },
    "transactions": {
        "quantity": {"min": 1, "max": 1000},
        "txn_date": {"format": "%Y-%m-%d"},
    },
    "external_metadata": {
        "sentiment_score": {"min": 0, "max": 1},
        "popularity_index": {"min": 0, "max": 10000},
        "last_updated": {"format": "%Y-%m-%d"},
    }
}

# ------------------------------------------------------------
# Helpers (pandas)
# ------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def check_schema(df: pd.DataFrame, expected_cols: list) -> dict:
    actual = list(df.columns)
    missing = [c for c in expected_cols if c not in actual]
    extra = [c for c in actual if c not in expected_cols]
    return {"missing_columns": missing, "extra_columns": extra}

def is_numeric_series(s: pd.Series) -> bool:
    try:
        pd.to_numeric(s.dropna())
        return True
    except Exception:
        return False

def range_check(s: pd.Series, min_v=None, max_v=None) -> dict:
    s_num = pd.to_numeric(s, errors="coerce")
    issues = []
    if min_v is not None:
        bad_min = s_num.dropna()[s_num.dropna() < min_v]
        if len(bad_min) > 0:
            issues.append(f"{len(bad_min)} values < {min_v}")
    if max_v is not None:
        bad_max = s_num.dropna()[s_num.dropna() > max_v]
        if len(bad_max) > 0:
            issues.append(f"{len(bad_max)} values > {max_v}")
    return {"issues": issues}

def format_check_date(s: pd.Series, fmt: str) -> dict:
    bad = []
    for i, v in s.dropna().items():
        try:
            datetime.strptime(str(v), fmt)
        except Exception:
            bad.append(i)
    return {"bad_count": len(bad), "bad_indices": bad[:10]}

def categorical_check(s: pd.Series, allowed: list) -> dict:
    invalid = s.dropna()[~s.dropna().isin(allowed)]
    return {"invalid_count": len(invalid), "examples": invalid.head(10).tolist()}

def duplicates_check(df: pd.DataFrame, key_cols: list) -> dict:
    dup = df.duplicated(subset=key_cols, keep=False)
    return {"duplicate_rows": int(dup.sum())}

def missing_values(df: pd.DataFrame) -> dict:
    mv = df.isna().sum()
    return {"missing_by_column": mv[mv > 0].to_dict(), "total_missing": int(mv.sum())}

def referential_integrity(child: pd.DataFrame, parent: pd.DataFrame, child_key: str, parent_key: str) -> dict:
    missing_refs = child[~child[child_key].isin(parent[parent_key])]
    return {"missing_ref_count": int(len(missing_refs)), "examples": missing_refs.head(10).to_dict(orient="records")}

# ------------------------------------------------------------
# Optional GE validation (modern API, guarded)
# ------------------------------------------------------------
def ge_validate(df: pd.DataFrame, dataset_name: str) -> dict:
    if not GE_ENABLED:
        return {"ge_enabled": False, "ge_success": None, "ge_statistics": {}, "ge_results": {}}

    try:
        engine = PandasExecutionEngine()
        validator = Validator(execution_engine=engine, batches=[{"batch_data": df}])

        def expect_not_null(col): validator.expect_column_values_to_not_be_null(col)
        def expect_unique(col): validator.expect_column_values_to_be_unique(col)
        def expect_between(col, min_v, max_v): validator.expect_column_values_to_be_between(col, min_value=min_v, max_value=max_v)
        def expect_in_set(col, allowed): validator.expect_column_values_to_be_in_set(col, allowed)
        def expect_date_fmt(col, fmt): validator.expect_column_values_to_match_strftime(col, fmt)

        if dataset_name == "product_catalog":
            expect_not_null("product_id"); expect_unique("product_id")
            expect_between("base_price", 0, 10000)
            expect_between("discount_percent", 0, 100)
            expect_between("avg_rating", 1, 5)
            expect_between("return_rate_percent", 0, 100)
            expect_between("profit_margin_percent", 0, 100)
            expect_in_set("is_perishable", ["Yes","No"])
            expect_date_fmt("launch_date", "%Y-%m-%d")

        elif dataset_name == "raw_customers":
            expect_not_null("customer_id"); expect_unique("customer_id")
            expect_between("age", 18, 100)
            expect_in_set("gender", ["M","F","O"])

        elif dataset_name == "raw_products":
            expect_not_null("product_id"); expect_unique("product_id")
            expect_not_null("product_name"); expect_not_null("category")

        elif dataset_name == "transactions":
            expect_not_null("txn_id"); expect_unique("txn_id")
            expect_not_null("customer_id"); expect_not_null("product_id")
            expect_between("quantity", 1, 1000)
            expect_date_fmt("txn_date", "%Y-%m-%d")

        res = validator.validate()
        stats = res.get("statistics", {})
        return {
            "ge_enabled": True,
            "ge_success": res.get("success"),
            "ge_statistics": stats,
            "ge_results": res,
        }
    except Exception as e:
        # If GE fails for any reason, continue with pandas-only results
        return {"ge_enabled": False, "ge_success": None, "ge_statistics": {"error": str(e)}, "ge_results": {}}

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
dfs = {}
for name, path in FILES.items():
    if os.path.exists(path):
        dfs[name] = load_csv(path)
    else:
        print(f"Warning: {path} not found.")

# Load partitioned transactions
import glob
txn_files = glob.glob(os.path.join(TXN_PATH, "**/*.csv"), recursive=True)
if txn_files:
    print(f"Loading {len(txn_files)} transaction partitions...")
    dfs["transactions"] = pd.concat([pd.read_csv(f) for f in txn_files], ignore_index=True)
else:
    print("Warning: No partitioned transactions found. Checking for legacy file...")
    legacy_txn = "raw_zone/recomart_raw_transactions_dec_2025.csv"
    if os.path.exists(legacy_txn):
        dfs["transactions"] = load_csv(legacy_txn)
    else:
        dfs["transactions"] = pd.DataFrame(columns=EXPECTED_SCHEMAS["transactions"]["columns"])

# ------------------------------------------------------------
# Validation pipeline
# ------------------------------------------------------------
summary = {}

for name, df in dfs.items():
    schema = check_schema(df, EXPECTED_SCHEMAS[name]["columns"])
    mv = missing_values(df)

    pk = None
    if name == "product_catalog":
        pk = ["product_id"]
    elif name == "raw_customers":
        pk = ["customer_id"]
    elif name == "raw_products":
        pk = ["product_id"]
    elif name == "transactions":
        pk = ["txn_id"]
    dup = duplicates_check(df, pk) if pk else {"duplicate_rows": None}

    rules = RULES.get(name, {})
    range_format_issues = {}

    for col, rule in rules.items():
        if col not in df.columns:
            range_format_issues[col] = {"error": "column_missing"}
            continue

        s = df[col]
        if "min" in rule or "max" in rule:
            if is_numeric_series(s):
                range_format_issues.setdefault(col, {})
                range_format_issues[col]["range"] = range_check(s, rule.get("min"), rule.get("max"))
            else:
                range_format_issues[col] = {"error": "non_numeric_values_present"}

        if "allowed" in rule:
            range_format_issues.setdefault(col, {})
            range_format_issues[col]["categorical"] = categorical_check(s, rule["allowed"])

        if "format" in rule:
            range_format_issues.setdefault(col, {})
            range_format_issues[col]["date_format"] = format_check_date(s, rule["format"])

    ge_res = ge_validate(df, name)

    summary[name] = {
        "row_count": int(len(df)),
        "schema": schema,
        "missing_values": mv,
        "duplicates": dup,
        "range_format_issues": range_format_issues,
        "ge": ge_res,
    }

# ------------------------------------------------------------
# Cross-dataset integrity checks
# ------------------------------------------------------------
integrity = {}
if "transactions" in dfs and "raw_customers" in dfs:
    integrity["txn_customer_fk"] = referential_integrity(
        dfs["transactions"], dfs["raw_customers"], "customer_id", "customer_id"
    )

if "transactions" in dfs and "product_catalog" in dfs:
    missing_in_catalog = dfs["transactions"][~dfs["transactions"]["product_id"].isin(dfs["product_catalog"]["product_id"])]
    integrity["txn_product_fk_catalog"] = {
        "missing_ref_count": int(len(missing_in_catalog)),
        "examples": missing_in_catalog.head(10).to_dict(orient="records"),
    }

if "external_metadata" in dfs and "product_catalog" in dfs:
    missing_meta = dfs["external_metadata"][~dfs["external_metadata"]["product_id"].isin(dfs["product_catalog"]["product_id"])]
    integrity["meta_product_fk"] = {
        "missing_ref_count": int(len(missing_meta)),
        "examples": missing_meta.head(10).to_dict(orient="records"),
    }

# ------------------------------------------------------------
# PDF Report
# ------------------------------------------------------------
class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Recomart Data Quality Report", ln=True, align="C")
        self.set_font("Arial", "", 10)
        self.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        self.ln(2)
        self.line(10, 28, 200, 28)
        self.ln(5)

    def section_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)

    def kv(self, key, value):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, f"{key}: {value}")

pdf = ReportPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Executive summary
pdf.section_title("Executive summary")
total_rows = sum(s["row_count"] for s in summary.values())
total_missing = sum(s["missing_values"]["total_missing"] for s in summary.values())
total_duplicates = sum((s["duplicates"]["duplicate_rows"] or 0) for s in summary.values())
pdf.kv("Datasets processed", ", ".join(summary.keys()))
pdf.kv("Total rows", total_rows)
pdf.kv("Total missing values (all datasets)", total_missing)
pdf.kv("Total duplicate rows (by primary keys)", total_duplicates)
pdf.kv("GE enabled", any(s["ge"]["ge_enabled"] for s in summary.values()))
pdf.ln(4)

# Dataset sections
for name, s in summary.items():
    pdf.section_title(f"Dataset: {name}")
    pdf.kv("Row count", s["row_count"])

    schema = s["schema"]
    pdf.kv("Missing columns", schema["missing_columns"] or "None")
    pdf.kv("Extra columns", schema["extra_columns"] or "None")

    mv = s["missing_values"]
    pdf.kv("Total missing values", mv["total_missing"])
    if mv["missing_by_column"]:
        pdf.kv("Missing by column", json.dumps(mv["missing_by_column"], indent=2))

    pdf.kv("Duplicate rows (by PK)", s["duplicates"]["duplicate_rows"])

    rf = s["range_format_issues"]
    if rf:
        pdf.kv("Range/format checks", json.dumps(rf, indent=2)[:2000])

    ge_stat = s["ge"]["ge_statistics"]
    pdf.kv("GE success", s["ge"]["ge_success"])
    pdf.kv("GE evaluated expectations", ge_stat.get("evaluated_expectations"))
    pdf.kv("GE successful expectations", ge_stat.get("successful_expectations"))
    pdf.kv("GE unsuccessful expectations", ge_stat.get("unsuccessful_expectations"))
    if "error" in ge_stat:
        pdf.kv("GE error", ge_stat["error"])
    pdf.ln(4)

# Cross-dataset integrity
pdf.section_title("Cross-dataset integrity checks")
for k, v in integrity.items():
    pdf.kv(k, json.dumps(v, indent=2)[:2000])
pdf.ln(4)

REPORT_NAME = "recomart_data_quality_report.pdf"
pdf.output(REPORT_NAME)

print(f"Report generated: {REPORT_NAME}")
print("\nSummary (compact):")
for name, s in summary.items():
    print(f"- {name}: rows={s['row_count']}, missing={s['missing_values']['total_missing']}, "
          f"dup={s['duplicates']['duplicate_rows']}, GE_enabled={s['ge']['ge_enabled']}, GE_success={s['ge']['ge_success']}")
print("\nCross-dataset integrity:")
for k, v in integrity.items():
    print(f"- {k}: {v['missing_ref_count'] if 'missing_ref_count' in v else 'see details'}")
