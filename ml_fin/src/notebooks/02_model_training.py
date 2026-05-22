# Databricks notebook source
import mlflow
import mlflow.spark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CreditRiskTrain").getOrCreate()

train_df = spark.table("hive_metastore.default.credit_train_dny")

feature_cols = ["log_income", "log_loan_amount", "credit_score_norm",
                "dti_ratio", "loan_income_ratio", "employment_years", "defaults_capped"]

mlflow.set_experiment("/Users/dnyandeepc18@gmail.com/credit_risk_model_training")

with mlflow.start_run(run_name="credit_risk_rf_v1") as run:
    rf = RandomForestClassifier(
        labelCol="default",
        featuresCol="scaled_features",
        numTrees=100,
        maxDepth=10,
        impurity="gini",
        seed=42
    )

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
    pipeline = Pipeline(stages=[assembler, scaler, rf])

    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 10)
    mlflow.log_param("impurity", "gini")
    mlflow.log_param("feature_count", len(feature_cols))

    model = pipeline.fit(train_df)

    mlflow.spark.log_model(model, "credit_risk_model")

    model.write().overwrite().save("/tmp/credit_risk_model")

    print(f"✅ Model trained and saved. Run ID: {run.info.run_id}")

run_info = mlflow.search_runs(experiment_ids=[run.info.experiment_id], order_by=["start_time DESC"], max_results=1)
display(run_info)