# Product-Sales-Analyst-for-Data-Driven-Insights
To create an advanced analytical system that provides deep insights into product sales, market trends, and customer behavior using a combination of data engineering, AI, machine learning, business analytics, and data analysis techniques.

# Data Ingestion and Engineering for Product Sales Analysis

This project implements a comprehensive ETL pipeline to ingest, clean, and transform sales, customer feedback, and competitor data for advanced product sales analysis.

## Data Sources

- Sales Data : Collected from online sales (e-commerce APIs), in-store sales (POS systems).
- Customer Feedback : Gathered from social media (Twitter API, Reddit API), and review sites (Yelp, Google Reviews).
- Competitor Data : Web scraped from competitor websites or gathered via public APIs for pricing and promotions.

## Technologies Used

- Apache Spark : For big data processing and ETL operations.
- AWS S3 : For scalable, durable storage of both raw and processed data.
- AWS Glue : For orchestration of ETL jobs.

## Pipeline Flow

1. Data Ingestion : Data is collected from various sources and stored in S3 raw data buckets.
2. ETL with Spark : Data is processed, cleaned, and transformed.
3. Output : Cleaned data is written back to S3 for further analysis or visualization.

## How to Run

1. Setup Environment :
   - `pip install -r requirements.txt`
   - Configure AWS credentials and ensure access to S3.

2. Ingest Data :
   - Run ingestion scripts in `ingestion_scripts/` to fetch data from APIs or scrape websites.

3. ETL Process :
   - Execute `main_etl_pipeline.py` to start the ETL process using Spark.

4. AWS Glue :
   - Use `glue_job.py` for scheduling and running ETL jobs in AWS environment.
