class classnet_model:

    # constructor:
    def __init__(self, df, input_features):

        # get unique outputs:
        outputs = set()
        point_arr = df.get_point_arr()
        output_row = df.get_col(len(point_arr[0]) - 1)
        for y in output_row:
            if y not in outputs:
                outputs.add(y)

        # calculate the mean numerical feature value:
        feature_map = df.get_feature_map()
        feature_mean = 0
        seen_features = 0

        for feature in input_features:
            col = df.get_col(feature_map[feature])
            for n in col:
                feature_mean += float(n)
                seen_features += 1
        
        feature_mean /= seen_features

        # determine output labels:
        step = feature_mean * 10
        label = step
        output_labels = {}
        for output in outputs:
            output_labels[output] = label
            label += step
        
        # calculate init weights per input feature:
        feature_weights = {}
        output_col = df.get_col(len(feature_map) - 1)
        for i in range(len(output_col)):
            output_col[i] = output_labels[output_col[i]]

        for feature in input_features:
            weight = 0
            feature_col = df.get_col(feature_map[feature])
            for i in range(len(feature_col)):
                if float(feature_col[i]) != 0:
                    weight += (float(output_col[i]) / float(feature_col[i]))
            weight /= len(feature_col)
            feature_weights[feature] = weight
        
        # testing - calculate percent error per feature:
        percent_error_map = {}
        for feature in feature_weights:
            percent_error = 0
            feature_col = df.get_col(feature_map[feature])
            for i in range(len(feature_col)):
                predicted = float(feature_weights[feature]) * float(feature_col[i])
                real = output_col[i]
                curr_error = (predicted - real) / real
                percent_error += curr_error
            percent_error /= len(feature_col)
            percent_error_map[feature] = percent_error
        
        print(percent_error_map)