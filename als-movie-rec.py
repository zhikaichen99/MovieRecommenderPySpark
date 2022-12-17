from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, LongType
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator 

import codecs
import sys

# numBlocks: used for parellization of computer. -1 implies autoconfigure
# rank: number of latent factors in model
# iterations: number of iterations to run. Use a grid search to optimize?
# lambda: regularization for the parameters
# ImplicitPreferences: specify whether we want to use feedback for the ALS. 
# alpha

def loadMovieNames():
    movieNames = {}
    
    with codecs.open('ml-1m/movies.dat', "r", encoding =  'ISO-8859-1', errors = 'ignore') as f:
        for line in f:
            fields = line.split('::')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

spark = SparkSession.builder.appName("ALSMovieRec").getOrCreate()

moviesSchema = StructType([
    StructField('userId', IntegerType(), True),
    StructField('movieId', IntegerType(), True),
    StructField('rating', IntegerType(), True),
    StructField('timeStamp', LongType(), True)
])

names = loadMovieNames()

data = spark.read.option('sep', '::').schema(moviesSchema).csv('ml-1m/ratings.dat')

# Splitting data into training and test set
(training, test) = data.randomSplit([0.7, 0.3], seed = 42)

# Setting ALS Model
USERID = 'userId'
MOVIE = 'movieId'
RATING = 'rating'

# This is the model after grid search and cross validation
als = ALS(
    maxIter = 5, 
    regParam = 0.01,
    rank = 10,
    nonnegative = True,
    implicitPrefs = False, 
    userCol = USERID, 
    itemCol = MOVIE, 
    ratingCol = RATING)


model = als.fit(data)

# Make predictions
predictions = model.transform(test)
predictions = predictions.na.drop()

# Evaluating model
eval = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
#rmse = eval.evaluate(predictions)
#print(f"RMSE: {rmse}")


# Making recommendations for a given user
# Construct a dataframe of the user ID's we want recommendations for
userID = int(sys.argv[1])
userSchema = StructType(
    [StructField("userId", IntegerType(), True)])
users = spark.createDataFrame([[userID,]], userSchema)

recommendations = model.recommendForUserSubset(users, 10)


print('Recommendations for User: ' + str(userID))

for userRecs in recommendations:
    myRecs = userRecs[1]
    for rec in myRecs:
        movie = rec[0] # Extracting movie ID
        rating = rec[1]
        movieName = names[movie]
        print(movieName + str(" ") +str(rating))




