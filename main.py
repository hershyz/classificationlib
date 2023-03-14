import dataframe
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet

df = dataframe.Dataframe('test-data/drug200.csv')
classnet_model = classnet.train(df, ['Age','Sex','BP','Cholesterol','Na_to_K'])