import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split


class sales_prediction:
    def __init__(self):
        pass
    def get_data(self):
        df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv")
        return df
    def drop_irrelevent_info(self, df):
        df1 = df.drop(columns='Unnamed: 0', axis=1)
        return df1
    def get_x_and_y(self, df):
        y = df['Sales']
        x = df.drop(columns='Sales')
        return x,y
    def get_scaling(self, x):
        scalar = StandardScaler()
        arr = scalar.fit_transform(x)
        return arr, scalar
    def train_data(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=100)
        elasticcv = ElasticNetCV(alphas=None, cv=5)
        elasticcv.fit(x_train, y_train)
        elastic = ElasticNet(alpha=elasticcv.alpha_, l1_ratio=elasticcv.l1_ratio_)
        elastic = elastic.fit(x_train, y_train)
        pickle.dump(elastic, open("ML2ndmodel", 'wb'))
        saved_maodel = pickle.load(open("ML2ndmodel", 'rb'))
        return saved_maodel
    def get_model_score(self, x,y):
        saved_model=pickle.load(open("ML2ndmodel", 'rb'))
        l = self.get_scaling(x)
        x = l[0]
        return saved_model.score(x,y)

    def get_prediction(self, data):
        df = self.get_data()
        df = self.drop_irrelevent_info(df=df)
        x,y = self.get_x_and_y(df=df)
        x, scalar =  self.get_scaling(x=x)
        data = scalar.transform(data)
        saved_model = pickle.load(open("ML2ndmodel",'rb'))
        return saved_model.predict(data)




