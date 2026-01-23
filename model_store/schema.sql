CREATE TABLE "co_occurrence" (
"product_a" TEXT,
  "product_b" TEXT,
  "frequency" INTEGER
);;
CREATE TABLE "features" (
"txn_id" TEXT,
  "txn_date" TEXT,
  "customer_id" TEXT,
  "product_id" TEXT,
  "quantity" INTEGER,
  "age" TEXT,
  "gender" TEXT,
  "product_name" TEXT,
  "super_category" TEXT,
  "category" TEXT,
  "brand" TEXT,
  "base_price" REAL,
  "discount_percent" INTEGER,
  "monthly_sales_volume" INTEGER,
  "avg_rating" REAL,
  "return_rate_percent" REAL,
  "profit_margin_percent" REAL,
  "is_perishable" TEXT,
  "shelf_life_days" INTEGER,
  "launch_date" TEXT,
  "sentiment_score" REAL,
  "popularity_index" REAL,
  "last_updated" TEXT,
  "product_price" REAL,
  "total_price" REAL,
  "user_txn_count" INTEGER,
  "prod_avg_quantity" REAL,
  "user_avg_score" REAL,
  "item_avg_score" REAL
);;
CREATE TABLE "interactions" (
"customer_id" TEXT,
  "product_id" TEXT,
  "interaction_score" INTEGER
);;
CREATE TABLE "product_metadata" (
"product_id" TEXT,
  "product_name" TEXT,
  "super_category" TEXT,
  "category" TEXT,
  "brand" TEXT,
  "base_price" REAL,
  "discount_percent" INTEGER,
  "monthly_sales_volume" INTEGER,
  "avg_rating" REAL,
  "return_rate_percent" REAL,
  "profit_margin_percent" REAL,
  "is_perishable" TEXT,
  "shelf_life_days" INTEGER,
  "launch_date" TEXT,
  "product_price" REAL,
  "sentiment_score" REAL,
  "popularity_index" REAL,
  "item_avg_score" REAL
);;
CREATE TABLE "products" (
"product_id" TEXT,
  "product_name" TEXT,
  "super_category" TEXT,
  "category" TEXT,
  "brand" TEXT,
  "base_price" INTEGER,
  "discount_percent" INTEGER,
  "monthly_sales_volume" INTEGER,
  "avg_rating" REAL,
  "return_rate_percent" REAL,
  "profit_margin_percent" REAL,
  "is_perishable" TEXT,
  "shelf_life_days" INTEGER,
  "launch_date" TEXT
);;
