import dataframe
import correlation
import line

df = dataframe.Dataframe('test-data/Stars.csv')
df.make_numerical()

x = df.get_col(3)
y = df.get_col(6)
L = line.Line(x, y)

correct = 0
for i in range(len(x)):
    prediction = round(L.predict(x[i]))
    real = y[i]
    if prediction == real:
        correct += 1
acc = correct / len(x)
print('acc: ' + str(acc))