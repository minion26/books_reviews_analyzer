#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Temporal Analysis (HDFS Version)
# This notebook analyzes genre growth trends from 2010 to 2015 using HDFS data.
# It also analyzes the popularity of different **Reading Formats** (E-Book, Audio, etc.) over time.
# 
# **Update**: Analysis is based on **Read Date** (`read_at`), not Publication Year.

# In[74]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, year, when, to_timestamp

# Set Plotting Style
sns.set_theme(style="whitegrid")

# Configurable Limit (Set to 0 for full dataset)
LIMIT = 0


# In[75]:


# Initialize Spark (Cluster Mode)
spark = SparkSession.builder \
    .appName("Goodreads_EDA_Temporal") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

print("✅ Spark Session created.")


# In[76]:


# HDFS Paths
hdfs_base = "hdfs:///user/ubuntu/goodreads_data/processed"
interactions_path = f"{hdfs_base}/master_interactions"


# In[77]:


# Read Master Interactions (Contains Book Metadata + User Ratings/Reads)
print("⏳ Reading master_interactions from HDFS...")
try:
    df = spark.read.parquet(interactions_path)
    print(f"✅ master_interactions read successfully.")
    print(f"Number of rows: {df.count()}")
    # Apply LIMIT for testing
    if LIMIT > 0:
        print(f"⚠️ Limiting analysis to {LIMIT} rows for testing.")
        df = df.limit(LIMIT)
        
    df.printSchema()
except Exception as e:
    print(f"❌ master_interactions not found or accessible: {e}")


# In[78]:


# Data Prep: Parse Dates & Filter
print("⏳ Processing Date Parsing...")

# Filter for READ books with valid read_at date
df_read = df.filter((col("is_read") == 1) & (col("read_at").isNotNull()))

# Parse 'read_at' string to Timestamp
# Format Example: 'Wed Jan 04 00:00:00 -0800 2012'
df_parsed = df_read.withColumn("read_date", to_timestamp(col("read_at"), "EEE MMM dd HH:mm:ss Z yyyy"))

# Extract Year
df_parsed = df_parsed.withColumn("read_year", year(col("read_date")))

# Filter for 2010-2015 based on READ YEAR
df_filtered = df_parsed.filter(
    (col("read_year") >= 2010) & 
    (col("read_year") <= 2015)
)

# Explode Shelves
df_exploded = df_filtered.select(
    col("read_year"), 
    explode(col("popular_shelves")).alias("shelf")
)

# Extract shelf name
df_raw_shelves = df_exploded.select(
    col("read_year"),
    col("shelf.name").alias("shelf_name")
)

print(f"✅ Valid Dated Interactions: {df_raw_shelves.count()}")


# ### Analysis A: Genre Trends (By Read Year)

# In[80]:


# Filter out common non-genre shelves
ignore_shelves = [
    'to-read', 'currently-reading', 'favorites', 'books-i-own', 
    'kindle', 'owned', 'ebook', 'library', 'to-buy', 'owned-books', 
    'audio', 'audiobook', 'ebooks', 'audiobooks', 'default', 'wish-list', 
    'my-books', 'read', 'borrowed', 'book-club', 'bookclub', 'book-group',
    'read-in-2010', 'read-in-2011', 'read-in-2012', 'read-in-2013', 'read-in-2014', 'read-in-2015',
    'read-2010', 'read-2011', 'read-2012', 'read-2013', 'read-2014', 'read-2015',
    'dnf', 'did-not-finish', 'abandoned', 'unfinished', 'stopped-reading', 'didn-t-finish',
    'adult', 'adult-fiction', 'novels', 'novel', 'books', 'hardcover', 'paperback',
    'series', 'trilogy', 'standalone', 're-read', 'reread', 'all-time-favorites',
    '5-stars', '4-stars', 'recommended', 'reviewed', 'netgalley', 'arc',
    'kindle-books', 'nook', 'audible', 'calibre', 'shelfari-favorites', 'favorites', 'favourites',
    'owned-tbr', 'tbr', 'to-be-read', 'my-library', 'library-books', 
    'free', 'freebie', 'giveaways', 'kindle-freebie', 'listened-to',
    'english', 'fiction', 'general-fiction', 'literature', 'e-book', 'e-books', 'maybe', 'own-it', 'on-hold', 'contemporary',
    'i-own', 'american', 'favorite'
]

df_genres = df_raw_shelves.filter(~col("shelf_name").isin(ignore_shelves))

# Smart Filters for partial matches (Exclude matches)
df_genres = df_genres.filter(~col("shelf_name").contains("read-in"))
df_genres = df_genres.filter(~col("shelf_name").contains("read-20")) # Catches read-2016, read-2017
df_genres = df_genres.filter(~col("shelf_name").contains("to-read")) # Catches to-read-non-fiction
df_genres = df_genres.filter(~col("shelf_name").contains("challenge"))
df_genres = df_genres.filter(~col("shelf_name").contains("kindle"))
df_genres = df_genres.filter(~col("shelf_name").contains("audio"))
df_genres = df_genres.filter(~col("shelf_name").contains("owned"))

# Normalize Genres
df_genres = df_genres.withColumn("genre", 
    when(col("shelf_name") == "nonfiction", "non-fiction")
    .otherwise(col("shelf_name"))
)

# Aggregation by READ YEAR
genre_counts = df_genres.groupBy("read_year", "genre").count().orderBy("read_year")

# Convert to Pandas
pdf_trends = genre_counts.toPandas()

# Find Top 10 Genres
if not pdf_trends.empty:
    top_genres = pdf_trends.groupby("genre")["count"].sum().nlargest(10).index.tolist()
    print(f"🔹 Top 10 Genres: {top_genres}")
    
    # Save CSV
    pdf_trends.to_csv("genre_trends_2010_2015.csv", index=False)
    
    # Filter Data for Top 10
    pdf_plot = pdf_trends[pdf_trends["genre"].isin(top_genres)]

    # Calculate Start vs End Stats
    print("\n🔹 Genre Counts: Start (2010) vs End (2015)")
    stats_df = pdf_plot[pdf_plot["read_year"].isin([2010, 2015])]
    if not stats_df.empty:
        stats_pivot = stats_df.pivot(index="genre", columns="read_year", values="count").fillna(0)
        # Handle cases where 2010 or 2015 might be missing columns if no data
        if 2010 not in stats_pivot.columns: stats_pivot[2010] = 0
        if 2015 not in stats_pivot.columns: stats_pivot[2015] = 0
        
        stats_pivot = stats_pivot[[2010, 2015]] # Reorder
        stats_pivot.columns = ["2010_Count", "2015_Count"]
        stats_pivot["Growth"] = stats_pivot["2015_Count"] - stats_pivot["2010_Count"]
        print(stats_pivot.sort_values("2015_Count", ascending=False))
    else:
        print("⚠️ Not enough data to compare 2010 vs 2015.")

    # Plot 1: Line Chart (Existing)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=pdf_plot, x="read_year", y="count", hue="genre", marker="o")
    plt.title("Genre Growth Trends (By Read Year)", fontsize=16)
    plt.ylabel("Books Read")
    plt.xlabel("Year Read")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Plot 2: Stacked Area Chart (New)
    print("🔹 Generating Stacked Area Chart...")
    # Pivot data: Year as Index, Genres as Columns
    pdf_pivot = pdf_plot.pivot(index="read_year", columns="genre", values="count").fillna(0)
    
    plt.figure(figsize=(12, 6))
    pdf_pivot.plot(kind='area', stacked=True, alpha=0.7, figsize=(12, 6), cmap="tab10")
    plt.title("Cumulative Genre Reading Trends (By Read Year)", fontsize=16)
    plt.ylabel("Total Books Read")
    plt.xlabel("Year")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ No genre data found.")


# ### Analysis B: Reading Format Trends
# Using the 'shelves' we ignored above to track formats (Kindle, Audio, etc.)

# In[7]:


from pyspark.sql.functions import lower

# Define Format Mappings
# Expanded list to include plurals and variations
format_keywords = [
    'kindle', 'ebook', 'e-book', 'e-books', 'ebooks', 'nook', 
    'audio', 'audiobook', 'audiobooks', 'audible', 'audio-book', 'audio-books',
    'hardcover', 'paperback', 'paperbacks'
]

df_formats = df_raw_shelves.filter(col("shelf_name").isin(format_keywords))

# We will map these specific shelves into broad Format Categories for a cleaner plot
df_formats_clean = df_formats.withColumn("format", 
    when(col("shelf_name").isin(["kindle", "nook", "ebook", "e-book", "e-books", "ebooks"]), "Digital (E-Book)")
    .when(col("shelf_name").isin(["audio", "audiobook", "audiobooks", "audible", "audio-book", "audio-books"]), "Audio")
    .when(col("shelf_name").isin(["hardcover", "paperback", "paperbacks"]), "Physical")
    .otherwise("Other") # Should have been filtered out by the list above, but for safety
)

# Aggregation
format_counts = df_formats_clean.groupBy("read_year", "format").count().orderBy("read_year")

# Convert to Pandas
pdf_formats = format_counts.toPandas()

if not pdf_formats.empty:
    print("🔹 Format Trends Data (Grouped):")
    print(pdf_formats.head())
    
    # Save CSV
    pdf_formats.to_csv("format_trends_2010_2015.csv", index=False)

    # Plot
    plt.figure(figsize=(12, 6))
    # Using 'lineplot' with markers for clear visibility
    sns.lineplot(data=pdf_formats, x="read_year", y="count", hue="format", style="format", markers=True, dashes=False, linewidth=2.5)
    plt.title("Reading Format Popularity (By Read Year)", fontsize=16)
    plt.ylabel("Books Read")
    plt.xlabel("Year Read")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ No format data found (shelves didn't match keywords).")

