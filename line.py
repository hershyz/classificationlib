import math

class Line:

    # constructor (y = a + bx)
    def __init__(self, x, y):

        self.x = x
        self.y = y

        sum_y = 0
        for n in y:
            sum_y += n
        
        sum_x = 0
        for n in x:
            sum_x += n

        sum_xsquared = 0
        for n in x:
            sum_xsquared += (n ** 2)

        sum_ysquared = 0
        for n in y:
            sum_ysquared += (n ** 2)
        
        sum_xy = 0
        for i in range(len(x)):
            sum_xy += (x[i] * y[i])
        
        n = len(x)
        self.a = ((sum_y * sum_xsquared) - (sum_x * sum_xy)) / ((n * sum_xsquared) - (sum_x ** 2))
        self.b = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_xsquared) - (sum_x ** 2))
    
    # display
    def display(self):
        print('y = ' + str(self.b) + 'x + (' + str(self.a) + ')')
    
    # predict
    def predict(self, x):
        return (self.b * x) + self.a
    
    # evaluate accuracy
    def eval(self):
        error = 0
        for i in range(len(self.x)):
            predicted = self.predict(self.x[i])
            diff = abs(predicted - self.y[i])
            error += (diff / self.y[i])
        error /= len(self.x)
        return 1 - error