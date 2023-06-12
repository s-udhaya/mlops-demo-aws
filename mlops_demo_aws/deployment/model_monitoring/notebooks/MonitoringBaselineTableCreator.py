# Databricks notebook source
env = dbutils.widgets.get("env")
model_name = dbutils.jobs.taskValues.get("Train", "model_name", debugValue="test-mlops-demo-aws-model")
model_version = dbutils.jobs.taskValues.get("Train", "model_version", debugValue="1")

# COMMAND ----------

MODEL_VERSION_COL = "served_model"
PREDICTION_COL = "predictions"  # What to name predictions in the generated tables
CATALOG_NAME = "udhay_demo"  # Name of the catalog to use
OUTPUT_SCHEMA_NAME = f"{env}_monitoring"  # Name of the output database to store tables in
BASELINE_TABLE = f"{model_name.replace('-', '_')}_baseline"

# COMMAND ----------

baseline_input_df = spark.table("udhay_demo.datasets.sf_airbnb_base_table")

# COMMAND ----------

import mlflow
from pyspark.sql.functions import struct, col, lit

model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="double")

baseline_input_df_with_pred =(baseline_input_df
                             .withColumn(PREDICTION_COL, loaded_model(struct(*map(col, baseline_input_df.columns))))
                             .withColumn(MODEL_VERSION_COL, lit(model_version)) )# Add model version column

display(baseline_input_df_with_pred)

# COMMAND ----------

(baseline_input_df_with_pred
 .write
 .format("delta")
 .mode("append")
 .option("delta.enableChangeDataFeed", "true")
 .saveAsTable(f"{CATALOG_NAME}.{OUTPUT_SCHEMA_NAME}.{BASELINE_TABLE}")
)
