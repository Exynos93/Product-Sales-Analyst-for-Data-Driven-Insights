import tweepy
import boto3
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FeedbackETL").getOrCreate()
s3 = boto3.client('s3')

def fetch_twitter_feedback():
    auth = tweepy.OAuthHandler("consumer_key", "consumer_secret")
    auth.set_access_token("access_token", "access_token_secret")
    api = tweepy.API(auth)
    
    feedback = []
    for tweet in tweepy.Cursor(api.search_tweets, q='your_brand', lang="en").items(100):
        feedback.append({"tweet_id": tweet.id, "text": tweet.text})

    return spark.createDataFrame(feedback)

def upload_feedback_to_s3(df, bucket, key):
    df.write.json(f"s3://{bucket}/{key}", mode="append")

if __name__ == "__main__":
    feedback_df = fetch_twitter_feedback()
    upload_feedback_to_s3(feedback_df, "your-feedback-bucket", "raw/feedback/")
