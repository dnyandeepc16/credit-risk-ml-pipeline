# Databricks notebook source
import mlflow
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("CreditRiskEval").getOrCreate()

model = PipelineModel.load("/tmp/credit_risk_model")
test_df = spark.table("hive_metastore.default.credit_test_dny")

predictions = model.transform(test_df)

evaluator_auc = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
evaluator_pr = BinaryClassificationEvaluator(labelCol="default", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="default", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="default", predictionCol="prediction", metricName="f1")

auc_roc = evaluator_auc.evaluate(predictions)
auc_pr = evaluator_pr.evaluate(predictions)
accuracy = evaluator_acc.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

tp = predictions.filter((col("default") == 1) & (col("prediction") == 1)).count()
fp = predictions.filter((col("default") == 0) & (col("prediction") == 1)).count()
fn = predictions.filter((col("default") == 1) & (col("prediction") == 0)).count()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

with mlflow.start_run(run_id=mlflow.active_run().info.run_id) if mlflow.active_run() else mlflow.start_run():
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("auc_pr", auc_pr)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

print("=" * 60)
print("  VALIDATION REPORT — Credit Risk Model")
print("=" * 60)
print(f"  AUC-ROC      : {auc_roc:.4f}  {'✅ PASS' if auc_roc >= 0.70 else '❌ FAIL'}")
print(f"  AUC-PR       : {auc_pr:.4f}  {'✅ PASS' if auc_pr >= 0.60 else '❌ FAIL'}")
print(f"  Accuracy     : {accuracy:.4f}  {'✅ PASS' if accuracy >= 0.75 else '❌ FAIL'}")
print(f"  Precision    : {precision:.4f}  {'✅ PASS' if precision >= 0.65 else '❌ FAIL'}")
print(f"  Recall       : {recall:.4f}  {'✅ PASS' if recall >= 0.60 else '❌ FAIL'}")
print(f"  F1 Score     : {f1:.4f}  {'✅ PASS' if f1 >= 0.65 else '❌ FAIL'}")
print("=" * 60)
print(f"  Confusion Matrix: TP={tp}  FP={fp}  FN={fn}")
print("=" * 60)