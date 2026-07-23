import pandas as pd
import re

# Load csv
df = pd.read_csv("all_results_description.csv")



systems = ["inkscape", "stackgres", "shepard"]
df = df[df["system"].isin(systems)]

has_precision = "precision_mean" in df.columns
has_recall = "recall_mean" in df.columns

# Model abbreviation mapping
model_map = {
    "feedforward_nn": "ffNN",
    "knn": "kNN",
    "naive_bayes": "NB",
    "logistic_regression": "LR",
    "random_forest": "RF",
    "xgboost": "XGB",
    "svm": "SVM",
    "sbert": "SB"
}

# Compute average F1 per configuration
avg_f1 = df.groupby(["detector", "model", "encoding"])["f1_mean"].mean().reset_index(name="avg_f1")

# Top-2 encodings per detector and model
top2 = (
    avg_f1.sort_values(["detector", "model", "avg_f1"], ascending=[True, True, False])
        .groupby(["detector", "model"])
        .head(2)
        .reset_index(drop=True)
)

# Merge with original df
merged = pd.merge(top2, df, on=["detector", "model", "encoding"], how="left")

# Pivot metrics
metrics = []
if has_precision:
    metrics.append("precision_mean")
if has_recall:
    metrics.append("recall_mean")
metrics.append("f1_mean")

pivot = merged.pivot_table(index=["detector", "model", "encoding"], 
                           columns="system", values=metrics)

# Flatten columns
pivot.columns = [f"{m}_{s}" for m, s in pivot.columns]

# Compute averages
if has_precision:
    pivot["avg_P"] = pivot[[f"precision_mean_{s}" for s in systems]].mean(axis=1)
else:
    pivot["avg_P"] = None

if has_recall:
    pivot["avg_R"] = pivot[[f"recall_mean_{s}" for s in systems]].mean(axis=1)
else:
    pivot["avg_R"] = None

pivot["avg_F1"] = pivot[[f"f1_mean_{s}" for s in systems]].mean(axis=1)

pivot = pivot.reset_index()

# Format function: three decimals, no leading zero
def fmt(x):
    if pd.isna(x):
        return "-"
    return f"{x:.3f}".lstrip("0")

# Encoding transformation function
def encode_latex(enc):
    enc_lower = enc.lower()
    # sbert_X → SB\textsubscript{X}
    match = re.match(r"sbert_(.+)", enc_lower)
    if match:
        return f"SB\\textsubscript{{{match.group(1)}}}"
    # tfidf → TF-IDF
    if enc_lower == "tfidf":
        return "TF-IDF"
    # openai → OpenAI
    if enc_lower == "openai":
        return "OpenAI"
    return enc

# Generate LaTeX rows separately per detector
for detector, group in pivot.groupby("detector"):
    print(f"% --- Detector: {detector} ---")
    for model, sub in group.groupby("model"):
        model_abbr = model_map.get(model, model)
        for i, row in sub.iterrows():
            # Use multirow for first encoding row
            if i == sub.index[0]:
                model_cell = f"\\multirow{{2}}{{*}}{{{model_abbr}}}"
            else:
                model_cell = ""
            
            # Skip Inkscape for detector 'arcan'
            row_systems = systems.copy()
            if detector.lower() == "arcan":
                row_systems.remove("inkscape")
            
            # Transform encoding
            encoding_latex = encode_latex(row['encoding'])
            
            latex_row = f"{model_cell} & {encoding_latex} & "
            
            for s in row_systems:
                if has_precision:
                    latex_row += f"{fmt(row.get(f'precision_mean_{s}'))} & "
                else:
                    latex_row += "- & "
                if has_recall:
                    latex_row += f"{fmt(row.get(f'recall_mean_{s}'))} & "
                else:
                    latex_row += "- & "
                latex_row += f"{fmt(row.get(f'f1_mean_{s}'))} & "
            
            # Average columns
            latex_row += f"{fmt(row['avg_P'])} & " if has_precision else "- & "
            latex_row += f"{fmt(row['avg_R'])} & " if has_recall else "- & "
            latex_row += f"{fmt(row['avg_F1'])} \\\\"
            
            print(latex_row)
        # Insert midrule after each model
        print("\\arrayrulecolor{kit-gray30}\\midrule\\arrayrulecolor{black}")
    print("\n")
