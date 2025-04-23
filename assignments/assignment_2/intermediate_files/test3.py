# %%
import os
import pandas as pd
import numpy as np 

# %%
RESULTS_DIR = "results_final"
OUTPUT_FLOWS_FILE = os.path.join(RESULTS_DIR, "flows.csv")

# %%
flows = pd.read_csv(OUTPUT_FLOWS_FILE)
print(f"Loaded {len(flows)} flows from '{OUTPUT_FLOWS_FILE}'")

# %%
# Remove classes with fewer than 10 instances
label_counts = flows["label"].value_counts()
labels_to_remove = label_counts[label_counts < 10].index
flows = flows[~flows["label"].isin(labels_to_remove)]

# Check the class distribution after filtering
print(flows["label"].value_counts())

# %%
print("#Unique Classes:", flows["label"].nunique())

# %%
print(flows.info())

# %%
print(flows.head())

# %%
print(flows.describe(include="all").T)

# %%
def get_IAT(timestamps):
    if len(timestamps) < 2:
        return []
    return [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]

# %%
def compute_stats(prefix, values):
    """Compute statistical features for a given list of values."""
    if not values:
        return {f"{prefix} {stat}": 0 for stat in [
            "Mean", "Median", "Std", "Min", "Max",
            "25%ile", "75%ile", "10%ile", "90%ile"
        ]}

    return {
        f"{prefix} Mean": np.mean(values),
        f"{prefix} Median": np.median(values),
        f"{prefix} Std": np.std(values) if len(values) > 1 else 0,
        f"{prefix} Min": np.min(values),
        f"{prefix} Max": np.max(values),
        f"{prefix} 25%ile": np.percentile(values, 25),
        f"{prefix} 75%ile": np.percentile(values, 75),
        f"{prefix} 10%ile": np.percentile(values, 10),
        f"{prefix} 90%ile": np.percentile(values, 90),
    }

# %%
def compute_flow_features(df):
    flow_features = []

    for _, row in df.iterrows():
        ts_list = eval(row['timestamps'])
        size_list = eval(row['sizes'])
        dir_list = eval(row['directions'])

        # Compute flow duration (assuming timestamps are in seconds)
        flow_duration = max(ts_list) - min(ts_list) if len(ts_list) >= 2 else 0

        # Separate Fwd and Bwd
        fwd_sizes = [s for s, d in zip(size_list, dir_list) if d == 1]
        bwd_sizes = [s for s, d in zip(size_list, dir_list) if d == 0]
        fwd_ts = [t for t, d in zip(ts_list, dir_list) if d == 1]
        bwd_ts = [t for t, d in zip(ts_list, dir_list) if d == 0]

        # Compute IATs
        flow_iat = get_IAT(ts_list)
        fwd_iat = get_IAT(fwd_ts)
        bwd_iat = get_IAT(bwd_ts)

        flow_data = {
            # Basic Flow-Level Features
            'Flow Duration': flow_duration,
            'Total Fwd Packets': len(fwd_sizes),
            'Total Bwd Packets': len(bwd_sizes),
            'Total Length of Fwd Packets': sum(fwd_sizes),
            'Total Length of Bwd Packets': sum(bwd_sizes),
        }

        # Add packet size and IAT stats
        flow_data.update(compute_stats("Fwd Packet Size", fwd_sizes))
        flow_data.update(compute_stats("Bwd Packet Size", bwd_sizes))
        flow_data.update(compute_stats("Fwd IAT", fwd_iat))
        flow_data.update(compute_stats("Bwd IAT", bwd_iat))

        flow_features.append(flow_data)

    return pd.DataFrame(flow_features)

# %%
df_features = compute_flow_features(flows)
final_df = pd.concat([flows.reset_index(drop=True), df_features.reset_index(drop=True)], axis=1)

print(final_df.shape)

# %%
# Print list of columns in the final DataFrame
print("Columns in the final DataFrame:")
for col in final_df.columns:
    print(f"- {col}")
# %%
final_df.columns
# %%
final_df.describe().T

# %%
df_features.columns
# %%
print("Flow Duration" in final_df.columns)

# %%
# Check for NaN values in the final DataFrame
data_tmp = final_df.copy()

# Replace infinities with NaN and drop rows with any NaNs in numeric cols
numeric_cols = data_tmp.select_dtypes(include=[np.number]).columns
data_tmp[numeric_cols] = data_tmp[numeric_cols].replace([np.inf, -np.inf], np.nan)
data_tmp.dropna(inplace=True)


# %%
from sklearn.preprocessing import LabelEncoder

# %%
# Encode the labels
le = LabelEncoder()
data_tmp["label_encoded"] = le.fit_transform(data_tmp["label"])

# %%
# Now extract features and labels
feature_cols = list(df_features.columns)
features = data_tmp[feature_cols]
labels = data_tmp["label_encoded"]

# %%
# Check the shapes of features and labels
print("Features shape:", features.shape)
print("Labels shape:", labels.shape)
# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simple Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_simple = rf.predict(X_test)
print(classification_report(y_test, y_pred_simple, target_names=le.classes_))
# %%
print(data_tmp["label"].value_counts())
# %%
from sklearn.metrics import accuracy_score

print("y_pred shape       : ", y_pred_simple.shape)                         # Predicted labels for test data
print()
print(f"Accuracy: {accuracy_score(y_test, y_pred_simple):.2f}")
# %%

# Confusion Matrix plot for simple RF
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

model = "simple RF"
cm = confusion_matrix(y_test, y_pred_simple)
plt.figure(figsize=(8, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix for {model}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# %%
# Plot for Feature Importance
# importances = rf.feature_importances_
importances = rf.feature_importances_
feature_names = features.columns

plt.figure(figsize=(6, 14))
bars = plt.barh(feature_names, importances, color='teal')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance")
for bar in bars:
    plt.text(bar.get_width() / 2,                               # X position (center of the bar)
             bar.get_y() + bar.get_height() / 2,                # Y position (center)
             f'{bar.get_width():.2f}',                          # Display importance value
             ha='center', va='center', fontsize=10, color='white')  
    
plt.show()
# %%
# Mapping encoded labels to class names
label_mapping_inv = {i: label for i, label in enumerate(le.classes_)}

# class distribution
unique_actual, counts_actual = np.unique(y_test, return_counts=True)    # Actual class distribution
unique_pred, counts_pred = np.unique(y_pred_simple, return_counts=True) # Predicted class distribution

# Convert numbers to class labels
actual_labels = [label_mapping_inv[i] for i in unique_actual]
predicted_labels = [label_mapping_inv[i] for i in unique_pred]

# Ensure all labels are present (even if missing in predictions)
all_labels = sorted(set(actual_labels) | set(predicted_labels), key=lambda x: x if x in label_mapping_inv else float('inf'))

# Convert to dictionary for alignment
actual_counts_dict = dict(zip(actual_labels, counts_actual))
predicted_counts_dict = dict(zip(predicted_labels, counts_pred))

# Fill missing classes with zero count
actual_counts = [actual_counts_dict.get(label, 0) for label in all_labels]
predicted_counts = [predicted_counts_dict.get(label, 0) for label in all_labels]

# Count correctly predicted samples for each class (True Positives)
true_positives = {label: 0 for label in all_labels}  # Initialize all as 0

for actual, pred in zip(y_test, y_pred_simple):
    if actual == pred:  # Correct prediction
        class_label = label_mapping_inv[actual]
        true_positives[class_label] += 1

# Convert true positives to a list for plotting
true_positive_counts = [true_positives.get(label, 0) for label in all_labels]

# Bar width and positions
x = np.arange(len(all_labels))  # Positions for bars
width = 0.3  # Width of bars

# Plot
plt.figure(figsize=(12,6))
bars1 = plt.bar(x - width, predicted_counts, width, label="Predicted Count", color="black", alpha=0.7)
bars2 = plt.bar(x, true_positive_counts, width, label="Correctly Predicted Count", color="blue", alpha=0.7)
bars3 = plt.bar(x + width, actual_counts, width, label="Actual Count", color="red", alpha=0.7)

# Add text labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height}', ha='center', va='bottom', fontsize=7, fontweight='bold')

# Labels and Title
plt.xlabel('Class Labels')
plt.ylabel('Count')
plt.title(f'Actual vs Predicted vs Correctly Predicted Class Distribution for Test Data Sized : {len(y_test)} samples')
plt.xticks(ticks=x, labels=all_labels, rotation=45, ha='right')  # Rotate x-labels for better readability
plt.legend()
plt.show()

# %%
from sklearn.model_selection import cross_val_score
import numpy as np

# Cross-validation for Random Forest
rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
rf_cv.fit(X_train, y_train)

# Perform 10-fold cross-validation and get accuracy scores
cv_scores = cross_val_score(rf, features, labels, cv=10, scoring='accuracy')

# Print the accuracy for each fold and the average accuracy
print(f"Cross-validation accuracies: {cv_scores}")
print(f"Average cross-validation accuracy: {np.mean(cv_scores)}")

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize the labels for multi-class ROC (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))  # Binarize the actual test labels
y_pred_prob = rf_cv.predict_proba(X_test)  # Get predicted probabilities for each class

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(y_test_bin.shape[1]):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)  # Calculate AUC for the class
    plt.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')

# Plotting the ROC curve (diagonal line)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Labels and Title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
# %%
