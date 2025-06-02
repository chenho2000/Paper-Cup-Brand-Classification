import json
import pandas as pd
from collections import Counter
from evidently.core.report import Report
from evidently.presets.drift import DataDriftPreset

with open("reference_log.json") as f:
    reference = json.load(f)
with open("prediction_log.json") as f:
    current = json.load(f)

# Convert to DataFrame

def flatten_hist(hist):
    # flattening color histograms
    return {f"{color}_bin_{i}": hist[color][i] for color in ['r', 'g', 'b'] for i in range(16)}

def log_to_df(log):
    rows = []
    for entry in log:
        row = {k: v for k, v in entry["input_image_stats"].items() if k != "color_histogram"}
        if "color_histogram" in entry["input_image_stats"]:
            row.update(flatten_hist(entry["input_image_stats"]["color_histogram"]))
        # Aggregate predictions
        preds = entry.get("predictions", [])
        row["num_predictions"] = len(preds)
        if preds:
            labels = [p.get("label") for p in preds]
            confidences = [p.get("confidence") for p in preds]
            row["most_common_label"] = Counter(labels).most_common(1)[0][0] if labels else None
            row["mean_confidence"] = sum(confidences) / len(confidences) if confidences else None
            label_dist = Counter(labels)
            row["label_count_timmies"] = label_dist.get("timmies", 0)
            row["label_count_paper_cup"] = label_dist.get("paper_cup", 0)
        else:
            row["most_common_label"] = None
            row["mean_confidence"] = None
            row["label_count_timmies"] = 0
            row["label_count_paper_cup"] = 0
        rows.append(row)
    return pd.DataFrame(rows)

ref_df = log_to_df(reference)
cur_df = log_to_df(current)

#check drift for all columns
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=ref_df, current_data=cur_df)
snapshot.save_html("drift_report.html")
print("Drift report saved as drift_report.html")