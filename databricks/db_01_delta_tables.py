# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Create Delta Tables from Medical Education Data
# MAGIC
# MAGIC Reads CSV files uploaded to Unity Catalog Volume and creates managed Delta tables
# MAGIC for the medical education RAG pipeline.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

CATALOG = "medical_education_rag_dbx"
SCHEMA = "rag_data"
VOLUME = "raw_data"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Using: {CATALOG}.{SCHEMA}")
print(f"Volume path: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Medical Chunks CSV and Create Delta Table

# COMMAND ----------

chunks_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(f"{VOLUME_PATH}/medical_chunks.csv")

print(f"Loaded {chunks_df.count()} chunks")
chunks_df.printSchema()
chunks_df.show(5, truncate=50)

# COMMAND ----------

chunks_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.medical_chunks")

print(f"Saved Delta table: {CATALOG}.{SCHEMA}.medical_chunks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Eval Queries and Create Delta Table

# COMMAND ----------

eval_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(f"{VOLUME_PATH}/eval_queries.csv")

print(f"Loaded {eval_df.count()} eval queries")
eval_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.eval_queries")

print(f"Saved Delta table: {CATALOG}.{SCHEMA}.eval_queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Load Test Queries and Create Delta Table

# COMMAND ----------

test_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(f"{VOLUME_PATH}/test_queries.csv")

print(f"Loaded {test_df.count()} test queries")
test_df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.test_queries")

print(f"Saved Delta table: {CATALOG}.{SCHEMA}.test_queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Delta Tables

# COMMAND ----------

tables = spark.sql(f"SHOW TABLES IN {CATALOG}.{SCHEMA}")
tables.show()

for table_name in ["medical_chunks", "eval_queries", "test_queries"]:
    count = spark.table(f"{CATALOG}.{SCHEMA}.{table_name}").count()
    print(f"  {table_name}: {count} rows")
