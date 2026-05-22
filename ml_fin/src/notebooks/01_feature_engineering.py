# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand, when, col, round, lit, expr
from datetime import datetime

spark = SparkSession.builder.appName("CreditRiskDataGen").getOrCreate()

spark.sql("DROP TABLE IF EXISTS hive_metastore.default.loan_applications_dny1")

df = spark.range(10000).select(
    (col("id") + 1001).alias("applicant_id"),
    round(rand() * 120 + 20, 2).alias("income"),
    round(rand() * 350 + 350, 0).alias("credit_score"),
    round(rand() * 50000 + 1000, 2).alias("loan_amount"),
    round(rand() * 60, 2).alias("dti_ratio"),
    round(rand() * 40, 0).alias("employment_years"),
    when(rand() < 0.3, 1).otherwise(0).alias("num_defaults")
)

df = df.withColumn(
    "default",
    when(
        (col("credit_score") < 580) | (col("dti_ratio") > 40) | (col("num_defaults") >= 1),
        1
    ).otherwise(0)
)

df.write.mode("overwrite").saveAsTable("hive_metastore.default.loan_applications_dny1")

display(spark.sql("SELECT COUNT(*) AS row_count FROM hive_metastore.default.loan_applications_dny1"))
display(spark.sql("""
    SELECT 
        ROUND(AVG(default) * 100, 2) AS default_rate_pct,
        SUM(default) AS total_defaults,
        COUNT(*) - SUM(default) AS total_non_defaults
    FROM hive_metastore.default.loan_applications_dny1
"""))


from pyspark.sql import SparkSession
from pyspark.sql.functions import log, col, when, round

spark = SparkSession.builder.appName("CreditRiskFeatureEng").getOrCreate()

df = spark.table("hive_metastore.default.loan_applications_dny1")

df_feat = df.select(
    col("applicant_id"),
    log(col("income")).alias("log_income"),
    log(col("loan_amount")).alias("log_loan_amount"),
    round((col("credit_score") - 350) / (850 - 350), 4).alias("credit_score_norm"),
    col("dti_ratio"),
    round(col("loan_amount") / col("income"), 4).alias("loan_income_ratio"),
    col("employment_years"),
    when(col("num_defaults") > 3, 3).otherwise(col("num_defaults")).alias("defaults_capped"),
    col("default")
).dropna()

train_df, test_df = df_feat.randomSplit([0.7, 0.3], seed=42)

train_df.write.mode("overwrite").saveAsTable("hive_metastore.default.credit_train_dny")
test_df.write.mode("overwrite").saveAsTable("hive_metastore.default.credit_test_dny")

display(spark.sql("SELECT COUNT(*) AS train_rows FROM hive_metastore.default.credit_train_dny"))
display(spark.sql("SELECT COUNT(*) AS test_rows FROM hive_metastore.default.credit_test_dny"))