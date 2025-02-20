# Databricks notebook source
# MAGIC %md
# MAGIC # Import Libraries

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.window import Window
from ifood_databricks import datalake, etl

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

# COMMAND ----------

# MAGIC %md
# MAGIC # Import tables into the Databricks environment

# COMMAND ----------

display(dbutils.fs.ls("/Volumes/tpcdi/tpcdi_raw_data/tpcdi_volume/case"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyzing the Data Profile

# COMMAND ----------

ab_test = spark.read.csv("/Volumes/tpcdi/tpcdi_raw_data/tpcdi_volume/case/ab_test_ref.csv", header=True, inferSchema=True)
display(ab_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ###  `teste AB`
# MAGIC - sem nulos
# MAGIC - somente Target e Control

# COMMAND ----------

user = spark.read.csv("/Volumes/tpcdi/tpcdi_raw_data/tpcdi_volume/case/consumer.csv.gz", header=True, inferSchema=True)
display(user)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `Users`
# MAGIC - Sem nulos

# COMMAND ----------

rest = spark.read.csv("/Volumes/tpcdi/tpcdi_raw_data/tpcdi_volume/case/restaurant.csv", header=True, inferSchema=True)
display(rest)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `Merchants`
# MAGIC - Sem nulls
# MAGIC - Ticket médio Max 100 (vamos usar isso na remoção de outliers)

# COMMAND ----------

orders = spark.read.json("/Volumes/tpcdi/tpcdi_raw_data/tpcdi_volume/case/order.json.gz")
display(orders)

# COMMAND ----------

# MAGIC %md
# MAGIC ### `Orders`
# MAGIC - remoção de custumer ids Nulos(0,23% das linhas)
# MAGIC - Tratamento de Outlier - Order total Amount 
# MAGIC

# COMMAND ----------

def calculate_iqr_outliers(df, column, factor=1.5):
    q1 = df.approxQuantile(column, [0.25], 0.05)[0]
    q3 = df.approxQuantile(column, [0.75], 0.05)[0]
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound

lower, upper = calculate_iqr_outliers(orders, "order_total_amount")

upper,lower
# Assumindo que o Lower é 0
lower = 0
upper,lower

# COMMAND ----------

outliers_orders = orders.filter((orders["order_total_amount"] < lower) | (orders["order_total_amount"] > upper))


data_without_outliers_orders = orders.filter(
    ((orders["order_total_amount"] >= lower) & (orders["order_total_amount"] <= upper)) | orders["order_total_amount"].isNull()
)

# COMMAND ----------

display(outliers_orders)

# COMMAND ----------

# removendo os custumoers ID sem id
data_without_outliers_orders_final = data_without_outliers_orders.filter(
    F.col("customer_id").isNotNull()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assembling the final table for the analysis

# COMMAND ----------

df_final_Q1 = (
    data_without_outliers_orders_final.join(user, ["customer_id"], "left")
    .join(ab_test, ["customer_id"], "inner")
    .join(rest,data_without_outliers_orders_final.merchant_id == rest.id , "left")
    .select(
        "customer_id",
        "merchant_id",
        "order_id",
        "order_total_amount",
        "order_created_at",
        "price_range",
        "average_ticket",
        "merchant_city",
        "merchant_state",
        "is_target",
    )
    .withColumn("order_created_at", F.to_date("order_created_at"))
    .withColumn("order_month_created_at", F.date_trunc("month", "order_created_at"))
)

# window func
window_spec = Window.partitionBy("customer_id").orderBy("order_created_at")

# rownumber
df_final_Q1 = df_final_Q1.withColumn("row_number", F.row_number().over(window_spec))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 1

# COMMAND ----------

first_base = (
    df_final_Q1.filter("row_number = 1")
    .groupBy("is_target")
    .agg(F.countDistinct("customer_id").alias("new_users"))
)
retained_base = (
    df_final_Q1.filter("row_number > 1")
    .groupBy("is_target")
    .agg(F.countDistinct("customer_id").alias("retained_users"))
)

stats = (
    first_base.join(retained_base, ["is_target"], "inner")
    .withColumn("not_retained", F.col("new_users") - F.col("retained_users"))
    .withColumn("%ret", F.col("retained_users") / F.col("new_users"))
)

display(stats)

# COMMAND ----------

# MAGIC %md
# MAGIC ### statistical validation

# COMMAND ----------

stats_with_array = stats.withColumn(
    "retention_array", 
    F.array("new_users","retained_users")
)

# Exibir o resultado
stats_with_array.select("is_target", "retention_array").show()

# COMMAND ----------

import numpy as np
from scipy.stats import chi2_contingency

# Criar uma tabela de contingência
tabela = np.array([[344666 , 256415],  # control
                   [428736 , 338607]]) # test

#  teste qui-quadrado
chi2, p, _, _ = chi2_contingency(tabela)

print("Estatística Qui-Quadrado:", chi2)
print("Valor P:", p)


if p < 0.05:
    print("Há uma diferença significativa na taxa de retenção entre Test/Control (p < 0.05).")
else:
    print("Não há diferença significativa na taxa de retenção entre Test/Control (p >= 0.05).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analisys

# COMMAND ----------

display(df_final_Q1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question 2

# COMMAND ----------

df_final_Q2 = (
    data_without_outliers_orders_final.join(user, ["customer_id"], "left")
    .join(ab_test, ["customer_id"], "inner")
    .join(rest,data_without_outliers_orders_final.merchant_id == rest.id , "left")
    .select(
        "customer_id",
        "merchant_id",
        "order_id",
        "order_total_amount",
        "order_created_at",
        "price_range",
        "average_ticket",
        "merchant_city",
        "merchant_state",
        "is_target",
    )
    .withColumn("order_created_at", F.to_date("order_created_at"))
    .withColumn("order_month_created_at", F.date_trunc("month", "order_created_at"))
)

# Define the window specification
window_spec = Window.partitionBy("customer_id","price_range").orderBy("order_created_at")

# Add the row_number column
df_final_Q2 = df_final_Q2.withColumn("row_number", F.row_number().over(window_spec))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Analysis

# COMMAND ----------

first_base_2 = (
    df_final_Q2.filter("row_number = 1")
    .groupBy("is_target","price_range")
    .agg(F.countDistinct("customer_id").alias("new_users"))
)
retained_base_2 = (
    df_final_Q2.filter("row_number > 1")
    .groupBy("is_target","price_range")
    .agg(F.countDistinct("customer_id").alias("retained_users"))
)

stats_2 = (
    first_base_2.join(retained_base_2, ["is_target","price_range"], "inner")
    .withColumn("not_retained", F.col("new_users") - F.col("retained_users"))
    .withColumn("%ret", F.col("retained_users") / F.col("new_users"))
)

display(stats_2)