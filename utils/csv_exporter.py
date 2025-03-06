import csv
import os
from datetime import datetime
from constants import Constants as K

def export_metrics(metrics, output_file=None):
    """_summary_

    Args:
        metrics (_type_): _description_
        otput_file (str, optional): _description_. Defaults to "exercise_metrics.csv".
    """
    if output_file is None:
        exercise_type = metrics.get("exercise", "exercise")
        output_file = f"./data_export/{exercise_type}_metrics.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    file_exists = os.path.isfile(output_file)

    final_metrics = {}
    
    for field in K.DEFAULT_COLUMNS_TO_EXPORT:
        if field == "exercise":
            pass
        value = metrics.get(field, '')
        if isinstance(value, float):
            value = round(value, 3)
        final_metrics[field] = value

    with open(output_file, mode='a', newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[f for f in K.DEFAULT_COLUMNS_TO_EXPORT if f!="exercise"])

        if not file_exists:
            writer.writeheader()
        writer.writerow(final_metrics)
