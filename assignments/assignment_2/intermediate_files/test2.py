# %%
import os
import pandas as pd

# %%
RESULTS_DIR = "resultsC"
OUTPUT_FLOWS_FILE = os.path.join(RESULTS_DIR, "flowsC.csv")

# %%
flows = pd.read_csv(OUTPUT_FLOWS_FILE)
print(f"Loaded {len(flows)} flows from '{OUTPUT_FLOWS_FILE}'")

# %%
print(flows.columns)

# %%
print(flows["label"].value_counts())
# %%

all_data = pd.read_csv("combined_data.csv") 
# %%
print(all_data.columns)
# %%
all_data["ip.src"].value_counts()
# %%
all_data["ip.dst"].value_counts()
# %%
