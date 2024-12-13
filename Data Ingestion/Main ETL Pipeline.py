from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SalesETL").getOrCreate()

# Load data from S3
sales_df = spark.read.parquet("s3a://your-sales-bucket/raw/sales/")
feedback_df = spark.read.json("s3a://your-feedback-bucket/raw/feedback/")
competitor_df = spark.read.json("s3a://your-competitor-data-bucket/raw/competitor/")

# Data Cleaning and Transformation
sales_df = sales_df.withColumn("date", to_date(col("timestamp"))).drop("timestamp")
feedback_df = feedback_df.withColumn("sentiment", when(col("text").contains("good"), "positive").otherwise("negative"))

# Join operations
sales_with_feedback = sales_df.join(feedback_df, sales_df.product_id == feedback_df.product_id, "left")

# Combine all data
final_df = sales_with_feedback.join(competitor_df, sales_with_feedback.product_name == competitor_df.name, "left")

# Normalize data types, handle missing values, etc.
final_df = final_df.na.fill("Unknown").withColumn("price", final_df.price.cast(DoubleType()))

# Write cleaned data back to S3
final_df.write.parquet("s3a://your-processed-data-bucket/processed/sales_feedback_competitor/")
