# MovieRecommenderPySpark
Movie Recommendations using Collaborative Filtering with Alternating Least Squares (ALS) in PySpark

#### -- Project Status: [Complete]

## Description

This module is designed to predict movie ratings for a user based on their past ratings and the ratings of other users. This project is built using the PySpark library and leverages pySpark's built  Alternating Least Squares (ALS) algorithm for collaborative filtering.

## Collaborative Filtering

Collaborative filtering is a technique used to make personalized recommendations by identifying patterns in a user's past behavior. In the context of this project, the past behavior we are interested in is the ratings that users have given to movies. By analyzing these ratings, the our model can identify other users who have given similar ratings to the same movies and use their ratings to make recommendations to the user.

## Alternating Least Squares (ALS)

The ALS algorithm is a matrix factorization technique that can be used to learn the underlying factors that influence the ratings given by users. It does this by decomposing the ratings matrix into two lower-dimensional matrices: one representing the preferences of the users and the other representing the characteristics of the movies. These matrices can then be used to make predictions about the ratings that a user would give to a particular movie.

## Evaluation Metric

The evaluation metric used is the Root Mean Squared Error (RMSE). The RMSE measures the difference between the predicted values and the actual values and takes the square root of the mean of the squared differences.

$$RMSE = \sqrt{\frac{\sum_{i=1} (\hat{y}_i - y_i)^2}{n}}$$

## Dataset

The dataset we are working with is the MovieLesn 1M dataset. It is a dataset of movie ratings collected by the GroupLens Research Project at the University of Minnesota. It contains 1 million ratings for approximately 3,900 movies made by 6,040 users. The ratings are on a scale of 1 to 5, with 1 being the lowest rating and 5 being the highest.

* *ratings.dat*: Contains the ratings data. Each line represents one rating. 
* *movies.dat*: Contains the movie data. Each line represents one movie.
* *users.dat*: Contains the user data. Each line represents one user. 

## Running Script

To get recommended movies for a specific user, run the *als-movie-rec.py* script in your terminal by executing the following: *spark-submit als-movie-rec.py 1* where 1 is the id of the user you want recommenddations for.


