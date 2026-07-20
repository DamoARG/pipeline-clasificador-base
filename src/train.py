import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Para datasets pequeños, limitar hilos evita sobrecarga innecesaria.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SIZE = 0.20


def set_seed(seed: int = SEED) -> None:
    """Fija semillas para que el experimento sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Configuración reproducible para CUDA.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Selecciona CUDA, MPS o CPU según disponibilidad."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_data(batch_size: int = BATCH_SIZE):
    """
    Carga Iris, divide entrenamiento/validación y estandariza las variables.

    El escalador se ajusta únicamente con los datos de entrenamiento para
    evitar filtración de información desde validación.
    """
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, X.shape[1], len(iris.target_names)


class IrisMLP(nn.Module):
    """MLP pequeño para clasificación multiclase del dataset Iris."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calcula la proporción de predicciones correctas."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    """Ejecuta una época completa de entrenamiento."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for features, targets in data_loader:
        features = features.to(device)
        targets = targets.to(device)

        # 1. Limpiar gradientes acumulados.
        optimizer.zero_grad()

        # 2. Forward pass.
        logits = model(features)

        # 3. Cálculo de la pérdida.
        loss = criterion(logits, targets)

        # 4. Backpropagation mediante autograd.
        loss.backward()

        # 5. Actualización de pesos.
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (
            torch.argmax(logits, dim=1) == targets
        ).sum().item()
        total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evalúa el modelo sin calcular ni acumular gradientes."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                torch.argmax(logits, dim=1) == targets
            ).sum().item()
            total_examples += batch_size

    average_loss = total_loss / total_examples
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def save_metrics(history: list[dict], output_dir: Path) -> None:
    """Guarda las métricas de cada época en formato CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict], output_dir: Path) -> None:
    """Genera y guarda las curvas de pérdida y accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_accuracy = [item["train_accuracy"] for item in history]
    val_accuracy = [item["val_accuracy"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Entrenamiento")
    plt.plot(epochs, val_loss, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Curva de pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracy, label="Entrenamiento")
    plt.plot(epochs, val_accuracy, label="Validación")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.title("Curva de accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()


def main() -> None:
    set_seed()

    device = get_device()
    print(f"Versión de PyTorch: {torch.__version__}")
    print(f"Dispositivo seleccionado: {device}")

    train_loader, val_loader, input_size, num_classes = load_data()

    model = IrisMLP(
        input_size=input_size,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2%}"
            )

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "outputs"

    save_metrics(history, output_dir)
    plot_history(history, output_dir)

    model_path = output_dir / "iris_mlp.pt"
    torch.save(model.state_dict(), model_path)

    final_metrics = history[-1]

    print("\nEntrenamiento finalizado.")
    print(f"Validation Loss final: {final_metrics['val_loss']:.4f}")
    print(f"Validation Accuracy final: {final_metrics['val_accuracy']:.2%}")
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas y gráficos guardados en: {output_dir}")


if __name__ == "__main__":
    main()
