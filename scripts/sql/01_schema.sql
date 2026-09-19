-- =====================================================================
-- Source system 1 of 3: the retailer's order-management system.
--
-- Design intent: this is the system of record for entities with a fixed,
-- well-understood shape. Primary keys make duplicates impossible and
-- foreign keys make dangling references impossible -- the database itself
-- enforces the contract, so downstream pipelines can trust it.
--
-- Load order matters: parents before children, or the FKs reject the rows.
-- =====================================================================

DROP TABLE IF EXISTS order_reviews  CASCADE;
DROP TABLE IF EXISTS order_payments CASCADE;
DROP TABLE IF EXISTS order_items    CASCADE;
DROP TABLE IF EXISTS orders         CASCADE;
DROP TABLE IF EXISTS customers      CASCADE;
DROP TABLE IF EXISTS products       CASCADE;
DROP TABLE IF EXISTS sellers        CASCADE;
DROP TABLE IF EXISTS geolocation    CASCADE;
DROP TABLE IF EXISTS category_translation CASCADE;

-- ---------- dimensions -------------------------------------------------

CREATE TABLE customers (
    customer_id              TEXT PRIMARY KEY,
    -- one physical person can hold several customer_ids (one per order);
    -- customer_unique_id is the real person. This distinction drives all
    -- of the RFM / churn work later, so it is called out explicitly here.
    customer_unique_id       TEXT NOT NULL,
    customer_zip_code_prefix INTEGER,
    customer_city            TEXT,
    customer_state           CHAR(2)
);
CREATE INDEX idx_customers_unique ON customers (customer_unique_id);

CREATE TABLE sellers (
    seller_id              TEXT PRIMARY KEY,
    seller_zip_code_prefix INTEGER,
    seller_city            TEXT,
    seller_state           CHAR(2)
);

CREATE TABLE category_translation (
    product_category_name         TEXT PRIMARY KEY,
    product_category_name_english TEXT NOT NULL
);

CREATE TABLE products (
    product_id                 TEXT PRIMARY KEY,
    -- deliberately NOT a foreign key to category_translation: the source
    -- translation file is incomplete, and two live categories have no
    -- English name. An FK here would block the load; instead the gap is
    -- caught as a data-quality finding and resolved in the silver layer.
    product_category_name      TEXT,
    product_name_length        INTEGER,
    product_description_length INTEGER,
    product_photos_qty         INTEGER,
    product_weight_g           NUMERIC,
    product_length_cm          NUMERIC,
    product_height_cm          NUMERIC,
    product_width_cm           NUMERIC
);
CREATE INDEX idx_products_category ON products (product_category_name);

CREATE TABLE geolocation (
    geolocation_id           BIGSERIAL PRIMARY KEY,   -- source has no key: many rows per zip
    geolocation_zip_code_prefix INTEGER NOT NULL,
    geolocation_lat          NUMERIC,
    geolocation_lng          NUMERIC,
    geolocation_city         TEXT,
    geolocation_state        CHAR(2)
);
CREATE INDEX idx_geo_zip ON geolocation (geolocation_zip_code_prefix);

-- ---------- facts ------------------------------------------------------

CREATE TABLE orders (
    order_id                      TEXT PRIMARY KEY,
    customer_id                   TEXT NOT NULL REFERENCES customers (customer_id),
    order_status                  TEXT NOT NULL,
    order_purchase_timestamp      TIMESTAMP NOT NULL,
    order_approved_at             TIMESTAMP,
    order_delivered_carrier_date  TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP
);
CREATE INDEX idx_orders_customer  ON orders (customer_id);
CREATE INDEX idx_orders_purchased ON orders (order_purchase_timestamp);
CREATE INDEX idx_orders_status    ON orders (order_status);

CREATE TABLE order_items (
    order_id           TEXT NOT NULL REFERENCES orders   (order_id),
    order_item_id      INTEGER NOT NULL,           -- line number within the order
    product_id         TEXT NOT NULL REFERENCES products (product_id),
    seller_id          TEXT NOT NULL REFERENCES sellers  (seller_id),
    shipping_limit_date TIMESTAMP,
    price              NUMERIC(12,2) NOT NULL CHECK (price >= 0),
    freight_value      NUMERIC(12,2)          CHECK (freight_value >= 0),
    PRIMARY KEY (order_id, order_item_id)
);
CREATE INDEX idx_items_product ON order_items (product_id);
CREATE INDEX idx_items_seller  ON order_items (seller_id);

CREATE TABLE order_payments (
    order_id             TEXT NOT NULL REFERENCES orders (order_id),
    payment_sequential   INTEGER NOT NULL,         -- an order can be split across instruments
    payment_type         TEXT,
    payment_installments INTEGER,
    payment_value        NUMERIC(12,2) CHECK (payment_value >= 0),
    PRIMARY KEY (order_id, payment_sequential)
);

CREATE TABLE order_reviews (
    -- review_id is NOT unique in the source file (a small number of reviews
    -- are duplicated across orders), so it cannot serve as the primary key.
    -- A surrogate key preserves every source row without silently dropping
    -- the duplicates -- they are reported by the data-quality layer instead.
    review_key              BIGSERIAL PRIMARY KEY,
    review_id               TEXT NOT NULL,
    order_id                TEXT NOT NULL REFERENCES orders (order_id),
    review_score            SMALLINT CHECK (review_score BETWEEN 1 AND 5),
    review_creation_date    TIMESTAMP,
    review_answer_timestamp TIMESTAMP
    -- NOTE: the free-text title/message columns are deliberately absent.
    -- Variable-length, frequently-absent, unstructured text belongs in the
    -- document store (source system 2), not in the relational core.
);
CREATE INDEX idx_reviews_order ON order_reviews (order_id);
CREATE INDEX idx_reviews_rid   ON order_reviews (review_id);
