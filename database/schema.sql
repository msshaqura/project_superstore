-- Dim tables
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT,
    segment TEXT
);

CREATE TABLE IF NOT EXISTS locations (
    location_id INT PRIMARY KEY,
    country TEXT,
    city TEXT,
    state TEXT,
    postal_code TEXT,
    region TEXT
);

CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    subcategory TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
    date_id DATE PRIMARY KEY,
    year INT,
    quarter INT,
    month INT,
    month_name TEXT,
    week INT,
    day INT,
    day_name TEXT,
    is_weekend BOOLEAN
);

-- Orders table (fact reference dim_date)
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    order_date DATE,
    ship_date DATE,
    ship_mode TEXT,
    shipping_time_days INT,
    date_id DATE REFERENCES dim_date(date_id)
);

-- Sales fact table
CREATE TABLE IF NOT EXISTS sales (
    id SERIAL PRIMARY KEY,
    order_id TEXT REFERENCES orders(order_id),
    customer_id TEXT REFERENCES customers(customer_id),
    product_id TEXT REFERENCES products(product_id),
    location_id INT REFERENCES locations(location_id),
    sales FLOAT,
    quantity INT,
    discount FLOAT,
    profit FLOAT,
    discount_bin TEXT
);
