#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


file_paths = {
    "HS2_export": "2010_2021_HS2_export.csv",
    "HS2_import": "2010_2021_HS2_import.csv",
    "old_export": "2018-2010_export.csv",
    "old_import": "2018-2010_import.csv"
}


# In[5]:


dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}


# In[6]:


exports = pd.concat([dataframes["HS2_export"], dataframes["old_export"]], ignore_index=True)
imports = pd.concat([dataframes["HS2_import"], dataframes["old_import"]], ignore_index=True)


# In[7]:


print("Exports Dataset Info:")
print(exports.info())


# In[11]:


print("Imports Dataset Info:")
print(imports.info())


# In[13]:


exports.head()


# In[15]:


imports.head()


# In[17]:


from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


def preprocess_data(df):
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    return df        


# In[20]:


exports = preprocess_data(exports)


# In[21]:


imports = preprocess_data(imports)


# In[25]:


scaler = StandardScaler()


# In[28]:


numeric_cols = exports.select_dtypes(include=["float64", "int64"]).columns


# In[31]:


exports[numeric_cols] = scaler.fit_transform(exports[numeric_cols])


# In[33]:


imports[numeric_cols] = scaler.fit_transform(imports[numeric_cols])


# In[35]:


X_exports = exports.drop(columns=[exports.columns[-1]])  
y_exports = exports[exports.columns[-1]]


# In[37]:


help(train_test_split)


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(X_exports, y_exports, test_size=0.2, random_state=42)


# In[41]:


from sklearn.ensemble import RandomForestRegressor


# In[42]:


model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[45]:


model.fit(X_train, y_train)


# In[46]:


y_pred = model.predict(X_test)


# In[47]:


from sklearn.metrics import mean_squared_error, r2_score


# In[48]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[49]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[50]:


results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})


# In[51]:


results.to_csv("export_prediction_results.csv", index=False)


# In[52]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# In[68]:


top_exports_per_year = exports.groupby("year")["Commodity"].value_counts().groupby(level=0).idxmax()


# In[72]:


print("Top Exported Commodities by Year:")
print(top_exports_per_year)


# In[98]:


plt.figure(figsize=(12, 6))
top_exports_plot = top_exports_per_year.apply(lambda x: x[1]) 
top_exports_plot.value_counts().plot(kind='bar', color='skyblue')
plt.title("Most Exported Commodities by Year")
plt.xlabel("Commodity")
plt.ylabel("Count of Years as Top Export")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()


# In[100]:


major_trade_commodity = exports["Commodity"].value_counts().idxmax()


# In[102]:


print(f"Major Trade Commodity: {major_trade_commodity}")


# In[104]:


plt.figure(figsize=(12, 6))
exports["Commodity"].value_counts().head(10).plot(kind='bar', color='orange')
plt.title("Top 10 Exported Commodities")
plt.xlabel("Commodity")
plt.ylabel("Export Volume")
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




