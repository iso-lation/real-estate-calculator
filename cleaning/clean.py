import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('../data/AmesHousing.csv')
# selecting useful columns
columns = ['Gr Liv Area', 'Bedroom AbvGr', 'Full Bath', 'Year Built', 'Garage Cars', 'Lot Area', 'SalePrice']
#making dataframe with selected columns
df = df[columns]
df = df.dropna()

# separate feautres (X) and target (y)
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
print("R2 score: ", r2_score(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

# save model
joblib.dump(model, '../models/price_estimator_model.pkl')

