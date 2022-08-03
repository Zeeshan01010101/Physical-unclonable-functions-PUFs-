from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import json
import time

spark = SparkSession.builder \
        .master("local[5]") \
        .appName("Full Model") \
        .config("spark.local.dir","/fastdata/acp21zgs/") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

data_train = spark.read.csv('../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv')

StringColumns = [x.name for x in data_train.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    data_train = data_train.withColumn(c, col(c).cast("double"))
data_train.printSchema()

update_func = (F.when(F.col('_c128') == -1, 0)
                .otherwise(F.col('_c128')))
data_train = data_train.withColumn('response', update_func)
data_train = data_train.drop("_c128")
data_train.show(5)
data_train.cache()

data_test = spark.read.csv('../Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv')


StringColumns = [x.name for x in data_test.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    data_test = data_test.withColumn(c, col(c).cast("double"))
data_test.printSchema()

update_func = (F.when(F.col('_c128') == -1, 0)
                .otherwise(F.col('_c128')))
data_test = data_test.withColumn('response', update_func)
data_test = data_test.drop("_c128")
data_test.show(5)
data_test.cache()


training_data_small, _ = data_train.randomSplit([0.01, 0.99], seed=123)

training_data_split, testing_data_split = training_data_small.randomSplit([0.7,0.3],seed = 123)
training_data_split.cache()

ncolumns = len(data_train.columns)

vecAssembler = VectorAssembler(inputCols=StringColumns[0:ncolumns-1], outputCol='challenges')
vecTrainingData = vecAssembler.transform(training_data_split)
vecTrainingData.select("challenges", "response").show(5)

#rf = RandomForestRegressor(labelCol="response", featuresCol="challenges",bootstrap = False)
rf = RandomForestClassifier(labelCol="response", featuresCol="challenges",bootstrap = False)

lr = LogisticRegression(featuresCol='challenges', labelCol='response', family='binomial')

layers = [len(data_train.columns)-1, 20, 5, 2]
mpc = MultilayerPerceptronClassifier(labelCol="response", featuresCol="challenges", layers=layers)

stages_rf = [vecAssembler, rf]
pipeline_rf = Pipeline(stages=stages_rf)

stages_lr = [vecAssembler, lr]
pipeline_lr = Pipeline(stages=stages_lr)

stages_nn = [vecAssembler, mpc]
pipeline_nn = Pipeline(stages=stages_nn)

paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .addGrid(rf.numTrees, [5, 8, 12]) \
    .addGrid(rf.featureSubsetStrategy, ['all','sqrt', 'log2']) \
    .build()

paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0.0, 0.2, 0.5, 0.7, 1.0]) \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
    .addGrid(lr.maxIter, [25, 50, 100]) \
    .build()

paramGrid_nn = ParamGridBuilder() \
    .addGrid(mpc.maxIter, [50, 100, 200]) \
    .addGrid(mpc.stepSize, [0.03, 0.01, 0.1]) \
    .addGrid(mpc.blockSize, [128,140,150]) \
    .build()

evaluator_M = MulticlassClassificationEvaluator\
      (labelCol="response", predictionCol="prediction", metricName="accuracy")

evaluator_B = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='response',metricName='areaUnderROC')

crossval_rf = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator_M,
                          numFolds=5)


crossval_lr = CrossValidator(estimator=pipeline_lr,
                          estimatorParamMaps=paramGrid_lr,
                          evaluator=evaluator_M,
                          numFolds=5)

crossval_nn = CrossValidator(estimator=pipeline_nn,
                          estimatorParamMaps=paramGrid_nn,
                          evaluator=evaluator_M,
                          numFolds=5)
start = time.process_time()
cvModel_rf = crossval_rf.fit(training_data_split)
print("Time required to fit Random Forest Model")
print(time.process_time() - start)
start = time.process_time()
prediction_rf = cvModel_rf.transform(testing_data_split)
print("Time required to transform Random Forest Model")
print(time.process_time() - start)
accuracy_rf = evaluator_M.evaluate(prediction_rf)

print("Accuracy for one percent rf model = %g " % accuracy_rf)

paramDict = {param[0].name: param[1] for param in cvModel_rf.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict, indent = 4))

start = time.process_time()
cvModel_lr = crossval_lr.fit(training_data_split)
print("Time required to fit Logistic Regression Model")
print(time.process_time() - start)
start = time.process_time()
prediction_lr = cvModel_lr.transform(testing_data_split)
print("Time required to transform Logistic Regression Model")
print(time.process_time() - start)
accuracy_lr = evaluator_M.evaluate(prediction_lr)

print("Accuracy for one percent lr model = %g " % accuracy_lr)

paramDict = {param[0].name: param[1] for param in cvModel_lr.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict, indent = 4))

start = time.process_time()
cvModel_nn = crossval_nn.fit(training_data_split)
print("Time required to fit Neural Networks Model")
print(time.process_time() - start)
start = time.process_time()
prediction_nn = cvModel_nn.transform(testing_data_split)
print("Time required to transform Neural Network Model")
print(time.process_time() - start)
accuracy_nn = evaluator_M.evaluate(prediction_nn)

print("Accuracy for one percent nn model = %g " % accuracy_nn)

paramDict = {param[0].name: param[1] for param in cvModel_nn.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict, indent = 4))

# Fitting and Transforming data on full dataset

#rf_full = RandomForestRegressor(labelCol="response", featuresCol="challenges",bootstrap = False)
rf_full = RandomForestClassifier(labelCol="response", featuresCol="challenges",bootstrap = False)

lr_full = LogisticRegression(featuresCol='challenges', labelCol='response', family='binomial')

layers_full = [len(data_train.columns)-1, 20, 5, 2]
mpc_full = MultilayerPerceptronClassifier(labelCol="response", featuresCol="challenges", layers=layers_full)

stages_rf_full = [vecAssembler, rf_full]
pipeline_rf_full = Pipeline(stages=stages_rf_full)

stages_lr_full = [vecAssembler, lr_full]
pipeline_lr_full = Pipeline(stages=stages_lr_full)

stages_nn_full = [vecAssembler, mpc_full]
pipeline_nn_full = Pipeline(stages=stages_nn_full)

start = time.process_time()
pipelineModel_lr = pipeline_lr_full.fit(data_train,cvModel_lr.bestModel.stages[-1].extractParamMap())
print("Time required to fit full Logistic Regression Model")
print(time.process_time() - start)
start = time.process_time()
prediction_lr_full = pipelineModel_lr.transform(data_test)
print("Time required to transform full Logistic Regression Model")
print(time.process_time() - start)
accuracy_lr_full = evaluator_M.evaluate(prediction_lr_full)
print("Accuracy for best lr model = %g " % accuracy_lr_full)
areaunderROC_lr_full = evaluator_B.evaluate(prediction_lr_full)
print("Area under ROC for best lr model = %g " % areaunderROC_lr_full)

start = time.process_time()
pipelineModel_rf = pipeline_rf_full.fit(data_train,cvModel_rf.bestModel.stages[-1].extractParamMap())
print("Time required to fit full Random Forest Model")
print(time.process_time() - start)
start = time.process_time()
prediction_rf_full = pipelineModel_rf.transform(data_test)
print("Time required to transform full Random Forest Model")
print(time.process_time() - start)
accuracy_rf_full = evaluator_M.evaluate(prediction_rf_full)
print("Accuracy for best rf model = %g " % accuracy_rf_full)
areaunderROC_rf_full = evaluator_B.evaluate(prediction_rf_full)
print("Area under ROC for best rf model = %g " % areaunderROC_rf_full)

start = time.process_time()
pipelineModel_nn = pipeline_nn_full.fit(data_train,cvModel_nn.bestModel.stages[-1].extractParamMap())
print("Time required to fit full Neural Network Model")
print(time.process_time() - start)
start = time.process_time()
prediction_nn_full = pipelineModel_nn.transform(data_test)
print("Time required to transform full Neural Network Model")
print(time.process_time() - start)
accuracy_nn_full = evaluator_M.evaluate(prediction_nn_full)
print("Accuracy for best nn model = %g " % accuracy_nn_full)
areaunderROC_nn_full = evaluator_B.evaluate(prediction_nn_full)
print("Area under ROC for best nn model = %g " % areaunderROC_nn_full)




