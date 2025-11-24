import dlt
from pyspark.sql.functions import *
from pyspark.sql.types import *

#CONFIGURATION
SOURCE_PATH = "/Volumes/workspace/taxi_assignment_db/taxi_raw_volume/"

# BRONZE LAYER (Ingestion)
# define valid column types to prevent string/int mismatches.

@dlt.table(
    comment="Raw ingestion of taxi data from the landing zone.",
    table_properties={"quality": "bronze"}
)
def taxi_bronze():
    return (
        spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("header", "true")
            .option("inferSchema", "true") 
            .load(SOURCE_PATH)
    )

# 2. SILVER LAYER (Cleaning & Transformation)
# Requirement: Drop non-positive trip_distance.
# Requirement: Flag suspicious rides.

@dlt.table(
    comment="Cleaned taxi data. Invalid distances dropped. Business logic applied.",
    table_properties={"quality": "silver"}
)
@dlt.expect_or_drop("valid_distance_rule", "trip_distance > 0")
def taxi_silver_clean():
    return (
        dlt.read_stream("taxi_bronze")
            #explicitly casting the cols to double to prevent sorting errors
            .withColumn("fare_amount", col("fare_amount").cast("double"))
            .withColumn("trip_distance", col("trip_distance").cast("double"))
            .withColumn(
                "price_per_mile", 
                round(col("fare_amount") / col("trip_distance"), 2)
            )
            # Logic: If price per mile > $50, suspicious
            .withColumn(
                "is_suspicious", 
                when(col("price_per_mile") > 50, True).otherwise(False)
            )
            .withColumn("week_start", date_trunc("week", col("timestamp")))
    )

# 3. SILVER LAYER (Aggregation)
# Requirement: Weekly aggregates.

@dlt.table(
    comment="Aggregated metrics by week."
)
def taxi_silver_weekly_stats():
    return (
        dlt.read_stream("taxi_silver_clean")
            .groupBy("week_start")
            .agg(
                count("*").alias("total_rides"),
                sum("fare_amount").alias("total_revenue"),
                avg("trip_distance").alias("avg_distance")
            )
    )

# 4. GOLD LAYER (Business Reporting)
# Requirement: Top 3 highest fare rides. (materialized view)

@dlt.table(
    comment="Executive report: Top 3 highest fares of all time."
)
def taxi_gold_top_fares():
    return (
        dlt.read("taxi_silver_clean")
            .select("ride_id", "timestamp", "passenger_id", "fare_amount", "is_suspicious")
            .orderBy(col("fare_amount").desc())
            .limit(3)
    )
