#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#we 
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# In[2]:


df = pd.read_csv("heart-disease.csv")
df


# In[3]:


df["target"].value_counts()


# In[4]:


df["target"].value_counts().plot(kind="bar", color=("magenta","orange"));


# In[5]:


df.sex.value_counts()


# In[6]:


pd.crosstab(df.target,df.sex).plot(kind="bar",
                                  figsize=(10,6),
                                  color=["salmon","lightblue"])
plt.title("Heart disease frequency for Sex")
plt.xlabel("0-Disease,1-disease")
plt.ylabel("amount")
plt.legend(["Female","Male"]);
plt.xticks(rotation=0);


# In[7]:


plt.figure(figsize=[10,6])
plt.scatter(df.age[df.target==1],
           df.thalach[df.target==1],
           c="salmon");
plt.scatter(df.age[df.target==0],
           df.thalach[df.target==0],
           c="lightblue");
plt.title("Heart disease in function of age and max Heart rate")
plt.xlabel("age")
plt.ylabel("Max heart rate")
plt.legend(["Heart disease","No heart disease"]);


# In[8]:


df.age.plot.hist();


# In[9]:


pd.crosstab(df.cp,df.target)


# In[10]:


pd.crosstab(df.cp,df.target).plot(kind="bar",
                                 figsize=(10,6),
                                 color=["salmon","lightblue"])
plt.title("Heart disease frequency per chest pain type")
plt.xlabel("Chest pain type")
plt.ylabel("Amount")
plt.legend(["Heart disease","No heart disease"]);
plt.xticks(rotation=0);


# In[11]:


df.corr()


# In[12]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt='.2f',
                 cmap="YlGnBu");


# In[13]:


X = df.drop("target", axis = 1)
Y = df["target"]


# In[14]:


X


# In[15]:


Y


# In[16]:


np.random.seed(42)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)


# In[17]:


Y_train, len(Y_train)


# In[18]:


models = {"Logistic Regression": LogisticRegression(),
         "KNN": KNeighborsClassifier(),
         "Random Forest": RandomForestClassifier()}

def fit_and_score(models, X_train, X_test, Y_train, Y_test):
    np.random.seed(42)
    model_scores = {}
    for name, model in models.items():
        
        model.fit(X_train,Y_train)
        
        model_scores[name] = model.score(X_test, Y_test)
        
    return model_scores


# In[19]:


model_scores = fit_and_score(models=models,
                            X_train=X_train,
                            X_test=X_test,
                            Y_train=Y_train,
                            Y_test=Y_test)

model_scores


# In[23]:


model_compare = pd.DataFrame(model_scores, index = ["accuracy"])
model_compare.T.plot.bar();


# In[20]:


train_scores = []
test_scores = []
neighbors = range(1,21)
knn = KNeighborsClassifier()
for i in neighbors:
    knn.set_params(n_neighbors=i)
    knn.fit(X_train,Y_train)
    train_scores.append(knn.score(X_train, Y_train))
    test_scores.append(knn.score(X_test, Y_test))


# In[21]:


train_scores


# In[22]:


test_scores


# In[23]:


plt.plot(neighbors, train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("No. of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on test data:{max(test_scores)*100:.2f}%")
      


# In[25]:


log_reg_grid = {"C": np.logspace(-4, 4,20),
                "solver":["liblinear"]}

rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth":[None, 3, 5, 10],
           "min_samples_split":np.arange(2, 20, 2),
           "min_samples_lead": np.arange(1 , 20, 2)}


# In[26]:


np.random.seed(42)

rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions = log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

rs_log_reg.fit(X_train, Y_train)


# In[27]:


rs_log_reg.best_params_


# In[28]:


rs_log_reg.score(X_test,Y_test)


# In[34]:


np.random.seed(42)

rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions = rf_grid,cv=5,
                           n_iter=20,
                           verbose=True)

rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth":[None, 3, 5, 10],
           "min_samples_split":np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1 , 20, 2)}

rs_rf.fit(X_train,Y_train)


# In[35]:


rs_rf.best_params_


# In[36]:


log_reg_grid = {"C": np.logspace(-4, 4, 30),"solver": ["liblinear"]}

gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose =True)

gs_log_reg.fit(X_train, Y_train);


# In[37]:


gs_log_reg.best_params_


# In[39]:


Y_preds = gs_log_reg.predict(X_test)


# In[40]:


Y_preds


# In[41]:


Y_test


# In[43]:


plot_roc_curve(gs_log_reg, X_test, Y_test);


# In[46]:


print(confusion_matrix(Y_test,Y_preds))


# In[56]:


sns.set(font_scale=1.5)
def plot_conf_mat(Y_test, Y_preds):
    fig, ax = plt.subplots(figsize=(3,3,))
    ax = sns.heatmap(confusion_matrix(Y_test, Y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("True Label")
    plt.ylabel("Predicted label")

plot_conf_mat(Y_test, Y_preds)


# In[57]:


print(classification_report(Y_test, Y_preds))


# In[58]:


gs_log_reg.best_params_


# In[59]:


clf = LogisticRegression(C=0.20433597178569418,
                        solver="liblinear")


# In[60]:


cv_acc = cross_val_score(clf,
                         X,
                         Y,
                         cv=5,
                         scoring="accuracy")

cv_acc


# In[61]:


cv_acc = np.mean(cv_acc)
cv_acc


# In[63]:


cv_precision = cross_val_score(clf,
                         X,
                         Y,
                         cv=5,
                         scoring="precision")

cv_precision = np.mean(cv_precision)
cv_precision


# In[64]:


cv_recall = cross_val_score(clf,
                         X,
                         Y,
                         cv=5,
                         scoring="recall")

cv_recall = np.mean(cv_recall)
cv_recall


# In[67]:


cv_f1 = cross_val_score(clf,
                         X,
                         Y,
                         cv=5,
                         scoring="f1")

cv_f1 = np.mean(cv_f1)
cv_f1


# In[70]:


cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall":cv_recall,
                           "F1": cv_f1},
                          index=[0])

cv_metrics.T.plot.bar(title="Cross-validated classification metrics",legend=False);


# In[74]:


clf = LogisticRegression(C=0.20433597178569418,
                        solver="liblinear")

clf.fit(X_train, Y_train);


# In[76]:


clf.coef_


# In[78]:


feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict


# In[80]:


feature_df = pd.DataFrame(feature_dict, index=[0])

feature_df.T.plot.bar(title="Feature Importance", legend=False);


# In[ ]:




