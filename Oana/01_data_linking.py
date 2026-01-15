#!/usr/bin/env python
# coding: utf-8

# # Phase 1: Data Linking (Optimized HDFS)
# This notebook joins Books, Interactions, and Reviews data using HDFS with optimized reads and early limiting.

# In[1]:


import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast, length
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, FloatType, ArrayType

# Configurable Limit (Set to 0 for full dataset)
LIMIT = 0


# In[2]:


# Initialize Spark Session (Cluster Mode)
spark = SparkSession.builder \
    .appName("Goodreads_Data_Linking") \
    .config("spark.driver.memory", "8g") \
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

print("✅ Spark Session created.")


# In[3]:


# HDFS Paths
hdfs_base = "hdfs:///user/ubuntu"
processed_dir = f"{hdfs_base}/goodreads_data/processed"

interactions_src = f"{hdfs_base}/goodreads_interactions_dedup.json.gz"
books_src = f"{hdfs_base}/goodreads_books.json.gz"
reviews_src = f"{hdfs_base}/goodreads_reviews_dedup.json.gz"


# In[4]:


# Explicit Schemas (CRITICAL for Performance on JSON)
# Defining schemas prevents Spark from scanning the entire file to infer types.

schema_books = StructType([
    StructField("book_id", StringType(), True),
    StructField("title", StringType(), True),
    StructField("average_rating", StringType(), True), # Often string in source, cast later
    StructField("publication_year", StringType(), True),
    StructField("publisher", StringType(), True),
    StructField("popular_shelves", ArrayType(StructType([
        StructField("count", StringType(), True),
        StructField("name", StringType(), True)
    ])), True)
])

schema_interactions = StructType([
    StructField("user_id", StringType(), True),
    StructField("book_id", StringType(), True),
    StructField("is_read", BooleanType(), True),
    StructField("rating", IntegerType(), True),
    StructField("read_at", StringType(), True),
    StructField("date_added", StringType(), True)
])

schema_reviews = StructType([
    StructField("user_id", StringType(), True),
    StructField("book_id", StringType(), True),
    StructField("review_text", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("n_votes", IntegerType(), True)
])


# In[5]:


# Read Books
print("⏳ Reading Books from HDFS...")
df_books = spark.read.schema(schema_books).json(books_src).select(
    col("book_id"), 
    col("title"), 
    col("average_rating").cast("float"),
    col("publication_year").cast("int"), 
    col("publisher"), 
    col("popular_shelves")
).filter(col("title").isNotNull())
print("✅ Finished reading books")


# In[6]:


# Read Interactions
print(f"⏳ Reading Interactions from HDFS (Limit: {LIMIT if LIMIT > 0 else 'FULL'})...")

# Using the Schema avoids the full file scan!
df_interactions_raw = spark.read.schema(schema_interactions).json(interactions_src)

# Apply LIMIT *before* filter to strictly limit file reading (Fastest)
if LIMIT > 0:
    df_interactions_raw = df_interactions_raw.limit(LIMIT)

df_interactions = df_interactions_raw.filter(
    (col("is_read") == True) | (col("rating") > 0)
).select(
    col("user_id"), 
    col("book_id"), 
    col("rating").cast("int"), 
    col("is_read").cast("int"),
    col("read_at"),
    col("date_added")
)

print("✅ Finished Reading Interactions")


# In[7]:


# Read Reviews
print("⏳ Reading Reviews from HDFS...")
df_reviews_raw = spark.read.schema(schema_reviews).json(reviews_src)

# Apply LIMIT *before* filter
if LIMIT > 0:
    df_reviews_raw = df_reviews_raw.limit(LIMIT)

df_reviews = df_reviews_raw.select(
    col("user_id"), 
    col("book_id"), 
    col("review_text"),
    col("rating").cast("int"), 
    col("n_votes").cast("int"),
).filter(length(col("review_text")) > 20)

print("✅ Finished Reading Reviews")


# In[8]:


print("🚀 Joining Data...")
# Remove broadcast(df_books). The books dataset is too large (2GB+) to broadcast,
# causing the Spark Driver to crash (OOM).
# Spark will automatically choose SortMergeJoin or ShuffleHashJoin.
master_interactions = df_interactions.join(df_books, on="book_id", how="inner")
master_reviews = df_reviews.join(df_books, on="book_id", how="inner")

out_inter = f"{processed_dir}/master_interactions"
out_rev = f"{processed_dir}/master_reviews"
    
print(f"💾 Saving to HDFS: {processed_dir}")
master_interactions.write.mode("overwrite").parquet(out_inter)
master_reviews.write.mode("overwrite").parquet(out_rev)

print("🎉 DONE! Processed data saved to HDFS.")


# In[ ]:




