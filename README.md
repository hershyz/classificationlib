![logo.png](logo.png)

Complete ML toolset for classification-based prediction problems.

### Install + Import:

[Release v1.0](https://github.com/hershyz/classificationlib/releases/tag/v1.0) (download and unzip)

```python
# importing all library modules:
import dataframe
import correlation_module
import sqrt_distance_classifier
import abs_distance_classifier
import percent_distance_classifier
import stddev_distance_classifier
import knn
import classnet
```

### Initializing Dataframe + Correlating Input Features to Output Feature:

```python
df = dataframe.Dataframe('test-data/drug200.csv')
correlation_module.run(df)
```

```
**output:**

highest correlation
---
Na_to_K
Cholesterol
BP
Sex
Age
---
lowest correlation
```

[Example CSV Format (Drug200 Dataset)](https://github.com/hershyz/classificationlib/blob/main/test-data/drug200.csv)

### Training + Predictions - Square Root Distance Classifier:

```python
 # (dataframe, input features)
model = sqrt_distance_classifier.train(df, ['Na_to_K', 'Cholesterol', 'BP'])
 # (model, dataframe, output feature)
print(sqrt_distance_classifier.eval(model, df, 'Drug')) # get accuracy
```

```python
**output:** 0.7
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, model)
print(sqrt_distance_classifier.predict(point_numerical, model))
```

```python
**output:** DrugY
```

### Training + Predictions - Absolute Distance Classifier:

```python
# (dataframe, input features)
model = abs_distance_classifier.train(df, ['Na_to_K', 'BP', 'Sex'])
# (model, dataframe, output feature)
print(abs_distance_classifier.eval(model, df, 'Drug')) # get accuracy
```

```python
**output:** 0.7
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, model)
print(abs_distance_classifier.predict(point_numerical, model))
```

```python
**output:** DrugY
```

### Training + Predictions - Percent Distance Classifier:

```python
# (dataframe, input features)
model = percent_distance_classifier.train(df, ['Na_to_K', 'BP', 'Age'])
# (model, dataframe, output feature)
print(percent_distance_classifier.eval(model, df, 'Drug')) # get accuracy
```

```python
**output:** 0.715
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, model)
print(percent_distance_classifier.predict(point_numerical, model))
```

```python
**output:** DrugY
```

### Training + Predictions - Standard Deviation Distance Classifier:

```python
# (dataframe, input features)
model = stddev_distance_classifier.train(df, ['Na_to_K', 'BP', 'Age'])
# (model, dataframe, output feature)
print(stddev_distance_classifier.eval(model, df, 'Drug')) # get accuracy
```

```python
**output:** 0.705
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, model)
print(stddev_distance_classifier.predict(point_numerical, model))
```

```python
**output:** DrugY
```

### Training + Predictions - ClassNet:

```python
# (dataframe, input features)
model = classnet.train(df, ['Na_to_K', 'Cholesterol', 'Sex', 'BP', 'Age'])
# (model, dataframe, output feature)
print(classnet.eval(model, df, 'Drug')) # get accuracy
```

```python
**output:** 0.24
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, model)
print(classnet.predict(point_numerical, model))
```

```python
**output:** DrugC
```

### Training + Predictions - KNN:

```python
df = dataframe.Dataframe('test-data/drug200.csv')
# (dataframe, input features, training dataset ratio)
model = knn.train(df, ['Na_to_K', 'Cholesterol', 'BP'], 'Drug', 0.5)
# (model, dataframe, output feature)
print(knn.eval(model, df, 'Drug'))
```

```python
**output:** 0.88
```

```python
point = {
    'Age': 23,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 25.355
}
point_numerical = df.convert_point(point)
# (converted numerical point, output feature, model)
print(knn.predict(point_numerical, 'Drug', model))
```

```python
**output:** DrugY
```