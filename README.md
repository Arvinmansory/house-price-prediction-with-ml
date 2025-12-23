import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("house.csv")

label_cols = ["Location","Condition","Garage"]


encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le 

X = df[["Bedrooms","Bathrooms","Floors","YearBuilt","Location","Condition","Garage"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mea = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mea}$')


new_house = {
    "Bedrooms": 3,     
    "Bathrooms": 2,      
    "Floors": 2, 
    "YearBuilt": 2000,
    "Location": "Urban",
    "Condition": "Fair",
    "Garage": "Yes"
}


for col in label_cols:
    le = encoders[col]
    
    if new_house[col] in le.classes_:
        new_house[col] = le.transform([new_house[col]])[0]
    else:
        new_house[col] = -1  

new_df = pd.DataFrame([new_house])

predicted_price = model.predict(new_df)[0]
print("🏠 Predicted House Price:", predicted_price)
