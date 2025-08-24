import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


pm = pd.read_csv("data/predictive_maintenance.csv")

pm['Failure Type'] = pm['Failure Type'].replace({'No Failure': 0,              # No Failure - 0     
                                                'Power Failure': 1,            # Power Failure - 1
                                                'Tool Wear Failure': 2,        # Tool Wear Failure - 2
                                                'Overstrain Failure': 3,       # Overstrain Failure - 3
                                                'Heat Dissipation Failure': 4, # Heat Dissipation Failure - 4
                                                'Random Failures': 5})         # Random Failures - 5

pm['Type'] = pm['Type'].replace({'L': 0,    # Type L(Low) - 0
                                 'M': 1,    # Type M(Medium) - 1
                                 'H': 2})   # Type H(High) - 2

# ------------------Data Normalization(Z-SCORE)-----------------------------
pm_cleaned = pm.drop(columns=['UDI', 'Product ID', 'Failure Type'])

pm_cleaned['Air temperature [K]'] = (pm_cleaned['Air temperature [K]'] - pm_cleaned['Air temperature [K]'].mean())/pm_cleaned['Air temperature [K]'].std()
pm_cleaned['Process temperature [K]'] = (pm_cleaned['Process temperature [K]'] - pm_cleaned['Process temperature [K]'].mean())/pm_cleaned['Process temperature [K]'].std()
pm_cleaned['Rotational speed [rpm]'] = (pm_cleaned['Rotational speed [rpm]'] - pm_cleaned['Rotational speed [rpm]'].mean())/pm_cleaned['Rotational speed [rpm]'].std()
pm_cleaned['Torque [Nm]'] = (pm_cleaned['Torque [Nm]'] - pm_cleaned['Torque [Nm]'].mean())/pm_cleaned['Torque [Nm]'].std()
pm_cleaned['Tool wear [min]'] = (pm_cleaned['Tool wear [min]'] - pm_cleaned['Tool wear [min]'].mean())/pm_cleaned['Tool wear [min]'].std()

#-------------------Training RANDOM FOREST CLASS WEIGHTING--------------------------------------- 
X = pm_cleaned.drop(columns=['Target']) # used for prediction
y = pm_cleaned['Target'] # what is to predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42, stratify=y)
#--------apply SMOTE to create synthetic failure cases and balance the classes---------------------------
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Make failures 50% of total
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

#-----------------------ACTUAL TRAINING-----------------------------------------
clf = RandomForestClassifier(    
    n_estimators=200,     # More trees
    max_depth=15,         # Limit depth to prevent overfitting
    min_samples_split=5,  # Minimum samples per split
    random_state=42
)
clf.fit(X_resampled, y_resampled)

#--------------------Evaluation------------------------------------
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

#-------------------FEATURE IMPORTANCE CHECK--------------------------
feature_importance = pd.Series(clf.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh', figsize=(10,5))
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Most Important Features")
plt.show()
