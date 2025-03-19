import pandas as pd

# Read the CSV file
df = pd.read_csv("output.csv")

# Add a column flagging mismatched alignment lengths
df["length_mismatch"] = df.apply(lambda row: len(row["qaln"]) != len(row["taln"]), axis=1)

# Print rows with mismatched lengths
mismatched_rows = df[df["length_mismatch"]]
if len(mismatched_rows) > 0:
    print("\nRows with mismatched alignment lengths:")
    print(mismatched_rows[["query", "target", "qaln", "taln", "length_mismatch"]])
    print(f"\nFound {len(mismatched_rows)} rows with mismatched lengths")
else:
    print("\nNo alignment length mismatches found")

# Save flagged results
df.to_csv("output_flagged.csv", index=False)
