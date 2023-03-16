import dataframe
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet
import correlation_module

'''
highest correlation
---
thall
caa
slp
oldpeak
exng
thalachh
restecg
fbs
chol
trtbps
cp
sex
age
---
lowest correlation
'''

df = dataframe.Dataframe('test-data/heart.csv')
model = classnet.train(df, ['thall', 'caa', 'slp', 'oldpeak', 'exng', 'thalachh', 'sex'])
print(classnet.eval(model, df, 'output'))