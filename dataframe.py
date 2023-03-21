class Dataframe:
    
    # constructor
    def __init__(self, path) -> None:
        
        # read raw file
        f = open(path)
        raw = f.readlines()

        lines = []
        # take newlines out of initial array
        for i in range(len(raw)):
            lines.append(raw[i].replace('\n', ''))
        
        # extract features, map to respective index
        features = lines[0].split(',')
        feature_map = {}
        for i in range(len(features)):
            feature_map[features[i]] = i
        self.feature_map = feature_map

        # extract dataframe into point array
        point_arr = []
        for i in range(1, len(lines)):
            point_arr.append(lines[i].split(','))
        self.point_arr = point_arr

        self.make_numerical()
    
    # display vanilla datastructures
    def display(self):
        print('feature index map: ' + str(self.feature_map))
        print('label serialization int map: ' + str(self.int_map))
        print('data:')
        for row in self.point_arr:
            print(row)
    
    # make categorical input features numerical
    def make_numerical(self):
        
        # generate int map
        n = 0
        int_map = {}
        for i in range(len(self.point_arr)):
            for j in range(len(self.feature_map) - 1):
                feature = self.point_arr[i][j]
                try:
                    float(feature)
                except: # if float conversion fails, the feature is a string
                    if feature not in int_map:
                        int_map[feature] = n
                        n += 1
        self.int_map = int_map

        # convert features to ints based on int map
        for row in range(len(self.point_arr)):
            for col in range(len(self.point_arr[row])):
                if self.point_arr[row][col] in self.int_map:
                    self.point_arr[row][col] = self.int_map[self.point_arr[row][col]]

    # returns a column index as a 1d array
    def get_col(self, index):
        res = []
        for row in self.point_arr:
            res.append(row[index])
        return res
    
    # getter for int map, must be called after make_numerical()
    def get_int_map(self):
        return self.int_map

    # getter for feature map
    def get_feature_map(self):
        return self.feature_map

    # get feature labels
    def get_feature_labels(self):
        feature_labels = []
        for label in self.feature_map:
            feature_labels.append(label)
        return feature_labels
    
    # get input feature labels
    def get_input_feature_labels(self):
        input_feature_labels = self.get_feature_labels()
        input_feature_labels.pop()
        return input_feature_labels

    # getter for point arr
    def get_point_arr(self):
        return self.point_arr
    
    # get test points
    def get_test_points(self):
        
        res = []
        point_arr = self.point_arr
        feature_map = self.feature_map

        for row in point_arr:
            map = {}
            for feature in feature_map:
                map[feature] = row[feature_map[feature]]
            res.append(map)
        
        return res
    
    # convert point
    def convert_point(self, point):
        int_map = self.int_map
        res = {}
        for key in point:
            if key in int_map:
                res[key] = int_map[key]
            else:
                res[key] = point[key]
        return res