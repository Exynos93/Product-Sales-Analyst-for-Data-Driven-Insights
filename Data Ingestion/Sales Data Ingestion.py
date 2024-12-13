import json
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Initialize Spark session
spark = SparkSession.builder.appName("SalesDataETL").getOrCreate()

# AWS S3 client for uploading data
s3 = boto3.client('s3')

def fetch_sales_data():
    # Example: Fetch from an e-commerce API or POS system
    # Here, we simulate data with a JSON file
    with open('data/sales_data.json', 'r') as f:
        sales_data = json.load(f)
    return spark.createDataFrame(sales_data)

def upload_to_s3(df, bucket, key):
    df.write.parquet(f"s3://{bucket}/{key}", mode="append")

if __name__ == "__main__":
    sales_df = fetch_sales_data()
    upload_to_s3(sales_df, "your-sales-bucket", "raw/sales/")
