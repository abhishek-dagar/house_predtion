import pandas as pd
data=pd.read_excel("housingdata.xlsx")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
corr_matrix=data.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scale',StandardScaler()),
])

x=my_pipeline.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#selecting a desired model for data
#from sklearn.linear_model import LinearRegression
#LR=LinearRegression()
from sklearn.ensemble import RandomForestRegressor
LR=RandomForestRegressor()
LR.fit(x_train,y_train)

from joblib import dump, load
dump(LR,"houseratepred.joblib")