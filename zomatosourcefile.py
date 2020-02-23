# Databricks notebook source
# MAGIC %md ## Loading Packages

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit, expr, avg
from pyspark.sql.functions import avg
from pyspark.sql.types import FloatType, IntegerType


# COMMAND ----------

# MAGIC %md ## Reading data set 
# MAGIC The data is read to a data frame with the header filed set to true, where the headers will also be read and the infer schema option is enabled where the data frame automatically infers the schema of different columns 

# COMMAND ----------

zomato_orgnl=spark.read.csv("/FileStore/tables/cleanedzomatodata.csv",header = 'True',inferSchema='True')

# COMMAND ----------

# MAGIC %md ## Printing Schema 
# MAGIC Below displayed is the schema of different columns inferred by the data frame

# COMMAND ----------

zomato_orgnl.printSchema()

# COMMAND ----------

# MAGIC %md ## Data Cleaning 

# COMMAND ----------

# MAGIC %md ### Counting null values
# MAGIC The null values present in different columns of the entire data is calculated using a function. Below provided is a function that counts the null values present according to columns

# COMMAND ----------

def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)

# COMMAND ----------

# MAGIC %md The function is called, where the parameter passed is the data frame. This allows the function to calculate the number of null values in each column of the data frame

# COMMAND ----------

null_columns_count_list = null_value_count(zomato_orgnl)


# COMMAND ----------

# MAGIC %md The count of null values are displayed according to columns

# COMMAND ----------

spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count']).show()

# COMMAND ----------

# MAGIC %md As the column dish_liked have huge number counts of null values, it is dropped to avoid further complexities in analysis

# COMMAND ----------

zomato_orgnl = zomato_orgnl.drop("dish_liked")

# COMMAND ----------

# MAGIC %md Printing the schema to see the rest available columns in the data frame

# COMMAND ----------

zomato_orgnl.printSchema()

# COMMAND ----------

# MAGIC %md The unique values of rate are displayed as column rate is the label and it is necessary to check whether there are any irrelevant values in that column

# COMMAND ----------

rate_new=zomato_orgnl.select("rate").dropDuplicates()
rate_new.show(70, False)

# COMMAND ----------

# MAGIC %md ### Filling Missisng Fields with Mean
# MAGIC As there are many null vaues in the columns rate and average cost, the mean values of both the columns are calculated to fill the empty filed in the column with the mean value of that column

# COMMAND ----------

mean_rate=zomato_orgnl.select([mean('rate')])
mean_rate.show()
mean_average_cost=zomato_orgnl.select([mean('average_cost')])
mean_average_cost.show()
zomato_orgnl = zomato_orgnl.withColumn("rate",when((zomato_orgnl["rate"].isNull()), 3.7).otherwise(zomato_orgnl["rate"]))
zomato_orgnl = zomato_orgnl.withColumn("average_cost",when((zomato_orgnl["average_cost"].isNull()), 561).otherwise(zomato_orgnl["average_cost"]))
zomato_orgnl.show(120)

# COMMAND ----------

# MAGIC %md ### Removing Irrelevant values
# MAGIC As there are values like `NEW` and `-` in the rate column, where the column should only have float values, those values are removed from the column

# COMMAND ----------

zomato_orgnl = zomato_orgnl[zomato_orgnl.rate != 'NEW']
zomato_orgnl = zomato_orgnl[zomato_orgnl.rate != '-']


# COMMAND ----------

# MAGIC %md After removing the irrelevant values and filling the missing fields with mean value, the unique values of column `rate` are displayed again to see whether there are any complexities

# COMMAND ----------

rate_new=zomato_orgnl.select("rate").dropDuplicates()
rate_new.show(70, False)

# COMMAND ----------

# MAGIC %md ### Type Conversion
# MAGIC The type of columns `rate` and `average cost` are inferred as strings, which are changed to type float

# COMMAND ----------

zomato_orgnl=zomato_orgnl.withColumn("rate", zomato_orgnl['rate'].cast(FloatType()))
zomato_orgnl=zomato_orgnl.withColumn("average_cost", zomato_orgnl['average_cost'].cast(FloatType()))
zomato_orgnl.printSchema()

# COMMAND ----------

# MAGIC %md ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md  The relation between location, average_cost and rating is visualized to check whether the rating and the avregae_cost of restaurants increases with respect to the area where the restaurant is located  

# COMMAND ----------

display(zomato_orgnl)

# COMMAND ----------

# MAGIC %md A visualization is created to check whether the rating of the restaurants increase when the restaurant provides an online ordering option

# COMMAND ----------

display(zomato_orgnl)

# COMMAND ----------

# MAGIC %md The relationship between the type of restaurant and the location is evaluated as the people living in some locality may only prefer restaurants of certain types

# COMMAND ----------

display(zomato_orgnl)

# COMMAND ----------

# MAGIC %md The relationship between `cuisines` and `location` is analyzed to check whether the people living in some localities only prefer a specific cuisine

# COMMAND ----------

display(zomato_orgnl)

# COMMAND ----------

# MAGIC %md The count of Restaurants according to the locality is provided below, which represents whether locality is an important factor for restaurants

# COMMAND ----------

numberofrestaurants = zomato_orgnl.groupBy("locality").count()
display(numberofrestaurants)

# COMMAND ----------

# MAGIC %md ### Feature Transformation
# MAGIC  As there are many categorical values in the data, which also cats as features for predicting the restaurant rating, the categorical columns are label indexed using `StringIndexer`and then encoded to binary vectors using `OneHotEncoderEstimator`. As all the features required for prediction need to be merged, `VectorAssembler` is used to merge all vectors along with the columns containing numerical values.

# COMMAND ----------

cols = zomato_orgnl.columns
categoricalColumns = ['online_order', 'book_table', 'rest_type', 'cuisines', 'locality']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index', handleInvalid = "keep")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'rate', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['votes', 'average_cost']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]    

# COMMAND ----------

# MAGIC %md Pipelines are used to perform feature transformation on the data frame

# COMMAND ----------

pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(zomato_orgnl)
zomato_orgnl = pipelineModel.transform(zomato_orgnl)
selectedCols = ['label', 'features'] + cols
zomato_orgnl = zomato_orgnl.select(selectedCols)


# COMMAND ----------

# MAGIC %md There is now a new `features` and `label` column appeared on the data frame after performing the feature transformation

# COMMAND ----------

zomato_orgnl.printSchema()

# COMMAND ----------

# MAGIC %md The entire data is now split to training and testing data where 70% of the entire data is used for training purpose and the rest 30% is used for testing

# COMMAND ----------

train, test = zomato_orgnl.randomSplit([0.7, 0.3], seed = 2018)
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

# COMMAND ----------

# MAGIC %md ##Models and Implementation

# COMMAND ----------

# MAGIC %md ### Logistic Regression
# MAGIC Applying Logistic Regression to train the model using the `train` data set and the trained model is applied for testing on the `test` data set 

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(train)
predictions = lrModel.transform(test)
predictions.select('average_cost', 'votes', 'label', 'rawPrediction', 'prediction', 'probability').show(10)

# COMMAND ----------

# MAGIC %md ###Accuracy Checking 
# MAGIC The accuracy of the predicted value is checked by determining the `Area Under ROC`Curve

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

# MAGIC %md ### Decision Tree
# MAGIC Applying Decision Tree to train the model using the `train` data set and the trained model is applied for testing on the `test` data set 

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
dtModel = dt.fit(train)
predictions = dtModel.transform(test)
predictions.select('average_cost', 'votes', 'label', 'rawPrediction', 'prediction', 'probability').show()


# COMMAND ----------

# MAGIC %md ###Accuracy Checking 
# MAGIC The accuracy of the predicted value is checked by determining the `Area Under ROC`Curve

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

# MAGIC %md ### Random Forest
# MAGIC Applying Random Forest to train the model using the `train` data set and the trained model is applied for testing on the `test` data set 

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
rfModel = rf.fit(train)
predictions = rfModel.transform(test)
predictions.select('average_cost', 'votes', 'label', 'rawPrediction', 'prediction', 'probability').show(10)



# COMMAND ----------

# MAGIC %md ###Accuracy Checking
# MAGIC The accuracy of the predicted value is checked by determining the `Area Under ROC`Curve

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))

# COMMAND ----------

# MAGIC %md It is evident from the above analysis that, Decision Tree performed better prediction when compared to Logistic Regression and Random Forest

# COMMAND ----------

# MAGIC %md ## Data Visualization

# COMMAND ----------

# MAGIC %md The data is visualized to find the count of restaurants that belongs to certain types 

# COMMAND ----------

restauranttypes = zomato_orgnl.groupBy("rest_type").count()
display(restauranttypes)

# COMMAND ----------

# MAGIC %md The data is visualized to see the number of restaurants that provide the facility to book a table in advance. It is evident from the visualization that, most restaurants of Banglore does not provide the facility of booking tables.

# COMMAND ----------

booktablefacility = zomato_orgnl.groupBy("book_table").count()
display(booktablefacility)

# COMMAND ----------

# MAGIC %md The data is displayed to show that the rating of the restaurant is dependant on online ordering option. It is evident from the below analysis that the restaurants that have online ordering options available hold a high rating when compared to the restaurants that does not have that facility.

# COMMAND ----------

group_data = zomato_orgnl.groupBy("online_order")
group_data.agg({'rate':'min'}).show()


# COMMAND ----------

# MAGIC %md The data is grouped based on average cost where the maximum ratings obtained by restaurants with different amount was determined

# COMMAND ----------

averagecost = zomato_orgnl.groupBy('average_cost')
averagecost.agg({'rate':'max'}).show()

# COMMAND ----------

# MAGIC %md ##Finding the Best Restaurant

# COMMAND ----------

# MAGIC %md ### Creating Temporary Table
# MAGIC Inorder to determine the best restaurants, various querying is performed on the data set for which the data needs to be displayed as an sql table where querying can be easily performed

# COMMAND ----------

temp_table_name="zomato"
zomato_orgnl.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md Displaying the temporary table created

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * from zomato

# COMMAND ----------

# MAGIC %md Finding the maximum and minimum limits of the average cost

# COMMAND ----------

max_average_cost=spark.sql("select max(average_cost) from `zomato` ")
min_average_cost=spark.sql("select min(average_cost) from `zomato` ")
max_average_cost.show()
min_average_cost.show()

# COMMAND ----------

# MAGIC %md Finding the maximum rating from the total ratings 

# COMMAND ----------

max_rate=spark.sql("select max(rate) from `zomato` ")
max_rate.show()

# COMMAND ----------

# MAGIC %md Finding restaurants that have a high rating and are cheaper where the average cost is less than 1500 Rupees

# COMMAND ----------

Cheapcosthighrating  = spark.sql("SELECT name,average_cost,rate,locality,rest_type,cuisines FROM `zomato`  WHERE rate>= 4.0 AND average_cost <1500 ORDER BY average_cost ASC ")
Cheapcosthighrating.show(50)

# COMMAND ----------

# MAGIC %md Finding the average votes obtained

# COMMAND ----------

mean_vote=spark.sql("select avg(votes) from `zomato` ")
mean_vote.show()

# COMMAND ----------

# MAGIC %md Finding the most reliable restaurants that have rating above `4.0`, votes above the `average votes` and cost below `1500 Rupees` which is 1/4th of the maximum expense for a meal of two people

# COMMAND ----------

reliablerestaurant =  spark.sql("SELECT name,average_cost,votes,rate,locality,rest_type,cuisines FROM `zomato`  WHERE rate>= 4.0 AND average_cost <1500 AND votes >= 297 ORDER BY average_cost ASC , rate DESC ")
reliablerestaurant.show()

# COMMAND ----------

# MAGIC %md Finding the most expensived restaurants that have rating above `4.2` and cost above `3000 Rupees` which is half of the maximum expense for a meal of two people

# COMMAND ----------

expensive_restaurants = spark.sql("SELECT name,average_cost,votes,rate,locality,rest_type,cuisines FROM `zomato`  WHERE rate>= 4.2 AND average_cost>3000  ORDER BY  rate DESC ")
expensive_restaurants.show()

# COMMAND ----------


