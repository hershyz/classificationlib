import dataframe
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet
import correlation_module

df = dataframe.Dataframe('test-data/drug200.csv')
correlation_module.run(df)