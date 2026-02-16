import pandas as pd
from db import engine
from sqlalchemy import text

def load_dim_tables(df, df_dim_date):
    # Customers
    df[['customer_id','customer_name','segment']].drop_duplicates().to_sql(
        'customers', engine, if_exists='replace', index=False
    )

    # Locations
    if 'location_id' not in df.columns:
        df['location_id'] = df.apply(
            lambda row: hash((row['city'], row['state'], row['postal_code'])) % 100000, axis=1
        )

    df[['location_id','country','city','state','postal_code','region']].drop_duplicates().to_sql(
        'locations', engine, if_exists='replace', index=False
    )


    # Products
    df[['product_id','product_name','category','subcategory']].drop_duplicates().to_sql(
        'products', engine, if_exists='replace', index=False
    )

    # Dim Date
    df_dim_date.to_sql('dim_date', engine, if_exists='replace', index=False)

def load_fact_tables(df):
    df_to_db = df[['order_id','customer_id','product_id','location_id',
                   'sales','quantity','discount','profit','order_date',
                   'ship_date','ship_mode']]

    # Orders
    df_orders = df[['order_id','order_date','ship_date','ship_mode']].drop_duplicates()
    df_orders['date_id'] = df_orders['order_date']
    df_orders.to_sql('orders', engine, if_exists='append', index=False)

    # Sales
    df_to_db.to_sql('sales', engine, if_exists='replace', index=False)
    print("✅ Tables chargées dans la DB")
