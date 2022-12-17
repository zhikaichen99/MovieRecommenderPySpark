# MovieRecommenderPySpark
Movie Recommendations using Collaborative Filtering with Alternating Least Squares (ALS) in PySpark

#### -- Project Status: [Active]

## Description

This module is designed to predict movie ratings for a user based on their past ratings and the ratings of other users. This project is built using the PySpark library and leverages pySpark's built  Alternating Least Squares (ALS) algorithm for collaborative filtering.

## Installation/Dependencies

## Collaborative Filtering

Collaborative filtering is a technique used to make personalized recommendations by identifying patterns in a user's past behavior. In the context of this project, the past behavior we are interested in is the ratings that users have given to movies. By analyzing these ratings, the system can identify other users who have given similar ratings to the same movies and use their ratings to make recommendations to the user.

## Alternating Least Squares (ALS)

The ALS algorithm is a matrix factorization technique that can be used to learn the underlying factors that influence the ratings given by users. It does this by decomposing the ratings matrix into two lower-dimensional matrices: one representing the preferences of the users and the other representing the characteristics of the movies. These matrices can then be used to make predictions about the ratings that a user would give to a particular movie.

## Evaluation Metric

## Dataset

