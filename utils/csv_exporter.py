import csv
import os
from datetime import datetime
from constants import Constants as K

# Variable global para almacenar el archivo de salida de la sesión actual.
SESSION_OUTPUT_FILE = None

def export_metrics(metrics, output_file=None):
    """
    Exporta las métricas de la repetición al CSV de la sesión actual.
    Si output_file es None, se utiliza una variable global para mantener el mismo
    archivo a lo largo de la sesión. Cuando se inicia un nuevo video, se debe reinicializar
    esta variable para generar un nuevo CSV.
    
    Args:
        metrics (dict): Diccionario con las métricas del ejercicio.
        output_file (str, optional): Ruta del archivo de salida. Si no se proporciona,
                                     se genera uno nuevo para la sesión actual.
    """
    global SESSION_OUTPUT_FILE

    exercise_type = metrics.get("exercise", "exercise")
    if output_file is None:
        if SESSION_OUTPUT_FILE is None:
            # Se genera el nombre del archivo con timestamp para la sesión actual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            SESSION_OUTPUT_FILE = f"./data_export/{exercise_type}/{timestamp}_{exercise_type}_metrics.csv"
        output_file = SESSION_OUTPUT_FILE

    # Se asegura de que la carpeta exista.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    final_metrics = {}
    for field in K.DEFAULT_COLUMNS_TO_EXPORT:
        if field == "exercise":
            continue
        value = metrics.get(field, '')
        if isinstance(value, float):
            value = round(value, 3)
        final_metrics[field] = value

    file_exists = os.path.isfile(output_file)
    # Se abre en modo 'a' para ir acumulando las métricas.
    with open(output_file, mode='a', newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[f for f in K.DEFAULT_COLUMNS_TO_EXPORT if f != "exercise"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(final_metrics)
