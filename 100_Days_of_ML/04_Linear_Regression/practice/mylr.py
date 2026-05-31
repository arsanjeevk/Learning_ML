class MyLinearRegression:

    def __init__(self):
        self.m = None
        self.b = None
        
    def fit(self, X_train, y_train):
        
        num = 0
        den = 0
        
        x_mean = X_train.mean()
        y_mean = y_train.mean()
        
        for i in range(X_train.shape[0]):
            num += (X_train.iloc[i] - x_mean) * (y_train.iloc[i] - y_mean)
            den += (X_train.iloc[i] - x_mean) ** 2
        
        self.m = num / den
        self.b = y_mean - (self.m * x_mean)
        
    def coef(self):
        return self.m
    
    def intercept(self):
        return self.b
    
    def predict(self, X_test):
        return self.m * X_test + self.b
    
