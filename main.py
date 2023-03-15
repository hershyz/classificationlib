import dataframe
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet

df = dataframe.Dataframe('test-data/heart.csv')
model = classnet.train(df, ['age','trtbps','exng','oldpeak','slp','caa','thall'])
print(classnet.eval(model, df, 'output'))