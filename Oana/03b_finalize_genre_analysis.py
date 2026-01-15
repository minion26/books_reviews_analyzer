#!/usr/bin/env python
# coding: utf-8

# # Phase 2: Genre Trends Cleanup & Visualization (Local)
# Processing the aggregated genre counts from the HDFS run. 
# filtering noise and generating final plots.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Load Data
df = pd.read_csv("genre_trends_2010_2015.csv")
print(f"Loaded {len(df)} genre-year records.")


# In[ ]:


# Cleanup Logic
ignore_list = ["have", "favorite-books", "books-i-own", "owned-books", "library", "default", "favorites"]
df_clean = df[~df["genre"].isin(ignore_list)].copy()

# Merge 'ya' -> 'young-adult'
df_clean["genre"] = df_clean["genre"].replace({"ya": "young-adult"})

# Re-aggregate (in case merge created duplicates)
df_clean = df_clean.groupby(["read_year", "genre"], as_index=False)["count"].sum()

# Get Top 10 Genres
top_genres = df_clean.groupby("genre")["count"].sum().nlargest(10).index.tolist()
print(f"🔹 Final Top 10 Genres: {top_genres}")

df_plot = df_clean[df_clean["genre"].isin(top_genres)]


# In[ ]:


# Statistics: Growth 2010 vs 2015
stats_df = df_plot[df_plot["read_year"].isin([2010, 2015])]
stats_pivot = stats_df.pivot(index="genre", columns="read_year", values="count").fillna(0)
if 2010 in stats_pivot.columns and 2015 in stats_pivot.columns:
    stats_pivot = stats_pivot[[2010, 2015]]
    stats_pivot["Growth"] = stats_pivot[2015] - stats_pivot[2010]
    stats_pivot["Growth %"] = (stats_pivot["Growth"] / stats_pivot[2010] * 100).round(1)
    print("\n🔹 Genre Growth (2010-2015):")
    print(stats_pivot.sort_values("Growth", ascending=False))


# In[ ]:


# Plot 1: Line Chart
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_plot, x="read_year", y="count", hue="genre", marker="o", linewidth=2.5)
plt.title("Genre Growth Trends by Read Year (2010-2015)", fontsize=16)
plt.ylabel("Books Read")
plt.xlabel("Year Read")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("genre_trends_line.png")
plt.show()


# In[ ]:


# Plot 2: Stacked Area Chart
df_pivot = df_plot.pivot(index="read_year", columns="genre", values="count").fillna(0)
plt.figure(figsize=(12, 6))
df_pivot.plot(kind='area', stacked=True, alpha=0.8, figsize=(12, 6), cmap="tab10")
plt.title("Cumulative Genre Composition (2010-2015)", fontsize=16)
plt.ylabel("Total Books Read")
plt.xlabel("Year")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("genre_trends_area.png")
plt.show()

