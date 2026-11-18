from singtown_ai import SingTownAIClient
from singtown_ai import stdout_watcher, file_watcher, error_watcher
from singtown_ai import export_yolo

client = SingTownAIClient()

log_watcher = stdout_watcher(interval=1)

@log_watcher
def on_stdout_write(content: str):
    client.log(content, end="")

@error_watcher()
def on_error():
    client.failed()

from ultralytics import YOLO
from pathlib import Path
import os
import shutil
import subprocess
import sys
from zipfile import ZipFile

DATASET_PATH = Path("../dataset")
RUNS_PATH = Path("../runs").absolute()
shutil.rmtree(RUNS_PATH, ignore_errors=True)
RUNS_PATH.mkdir(parents=True)
METRICS_PATH = RUNS_PATH / "train/results.csv"

@file_watcher(METRICS_PATH, interval=3)
def file_on_change(content: str):
    import csv
    from io import StringIO

    metrics = list(csv.DictReader(StringIO(content)))
    if not metrics:
        return
    client.update_metrics(metrics)

LABELS = client.task.project.labels
MODEL_NAME = client.task.model_name
EPOCHS = client.task.epochs
BATCH_SIZE = client.task.batch_size
LEARNING_RATE = client.task.learning_rate
EXPORT_WIDTH = client.task.export_width
EXPORT_HEIGHT = client.task.export_height

MODEL_CLASS, IMG_SZ = MODEL_NAME.split("_")

def export_tflite(weights: Path) -> Path:
    """Export the trained weights in a child process that cannot see the GPU.

    With CUDA visible, ultralytics wants onnxruntime-gpu for the export and
    pip-installs it in the middle of the task, and TensorFlow runs the INT8
    conversion on the GPU. Hiding the GPU keeps the export on the CPU path,
    the same one a machine without a GPU takes.
    """
    process = subprocess.Popen(
        [sys.executable, "export.py", str(weights), str(EXPORT_HEIGHT), str(EXPORT_WIDTH)],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
    )
    for raw_line in process.stdout:
        # A progress bar redraws itself with "\r": keep only its final state.
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n").rsplit("\r", 1)[-1]
        # stdout_watcher only replaces this process's sys.stdout, so the child's
        # output is printed here to reach the task log. The watcher reads then
        # truncates its buffer; hold its lock so a line written between the
        # two is not lost.
        with log_watcher.lock:
            print(line)
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)
    return weights.with_name(f"{weights.stem}_saved_model") / f"{weights.stem}_integer_quant.tflite"

print("Download dataset")
export_yolo(client, DATASET_PATH)

model = YOLO("yolov8n.pt")
results = model.train(data="../dataset/data.yaml", epochs=EPOCHS, imgsz=IMG_SZ, batch=BATCH_SIZE, project=str(RUNS_PATH))
print(results)

# The checkpoint model.export() would use: the best epoch, or the last one if
# no validation ran.
weights = model.trainer.best if model.trainer.best.exists() else model.trainer.last
export_path = export_tflite(Path(weights))
print(f"Exported model to {export_path}")

with open(RUNS_PATH/"trained.txt", "wb") as f:
    f.write("\n".join(LABELS).encode("utf-8"))

with ZipFile(RUNS_PATH/"result.zip", 'w') as zip:
    zip.write("openmv_yolov8n.py", arcname="main.py")
    zip.write(export_path, arcname="trained.tflite")
    zip.write(RUNS_PATH/"trained.txt", arcname="trained.txt")

client.upload_results_zip(RUNS_PATH/"result.zip")
client.success()
print("Finished")
