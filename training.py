# ============================================================
# training.py
#
# Módulo de entrenamiento y evaluación para el Transformer
# ============================================================

import time
import inspect
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix


def _forward_pass(model, patches, endmembers_tensor):
    """
    Función auxiliar para adaptar el pase forward dinámicamente según 
    si el modelo requiere 1 o 2 entradas.
    """
    # Verificar cuántos argumentos requiere la función forward del modelo
    sig = inspect.signature(model.forward)
    num_params = len(sig.parameters)

    if num_params >= 2:
        return model(patches, endmembers_tensor)
    else:
        return model(patches)


def train_epoch(model, dataloader, endmembers_tensor, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for patches, labels in dataloader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = _forward_pass(model, patches, endmembers_tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * patches.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, endmembers_tensor, device, ignore_index=-100):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patches, labels in dataloader:
            patches = patches.to(device)
            logits = _forward_pass(model, patches, endmembers_tensor)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_targets.append(labels.numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Filtrado dinámico: Seleccionar solo píxeles distintos del ignore_index (-100)
    valid_mask = all_targets != ignore_index
    valid_preds = all_preds[valid_mask]
    valid_targets = all_targets[valid_mask]

    if len(valid_targets) == 0:
        return 0.0, 0.0, 0.0, np.array([])

    unique_labels = np.sort(np.unique(valid_targets))

    oa = accuracy_score(valid_targets, valid_preds)
    f1 = f1_score(valid_targets, valid_preds, labels=unique_labels, average="macro", zero_division=0)
    kappa = cohen_kappa_score(valid_targets, valid_preds, labels=unique_labels)
    cm = confusion_matrix(valid_targets, valid_preds, labels=unique_labels)

    return oa, f1, kappa, cm


def run_training_pipeline(model, train_loader, test_loader, endmembers_tensor, epochs, lr, device, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n" + "=" * 70)
    print(f"INICIANDO ENTRENAMIENTO ({type(model).__name__}) - {epochs} ÉPOCAS")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, endmembers_tensor, criterion, optimizer, device)
        elapsed = time.time() - start_time

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Época [{epoch:03d}/{epochs:03d}] - Loss: {train_loss:.6f} | Tiempo: {elapsed:.2f}s")

    print("\n" + "=" * 70)
    print("EVALUACIÓN EN IMAGEN DE ENTRENAMIENTO (TRAIN)")
    print("=" * 70)
    train_oa, train_f1, train_kappa, _ = evaluate(model, train_loader, endmembers_tensor, device)
    print(f"Train Overall Accuracy (OA): {train_oa * 100:.2f}%")

    print("\n" + "=" * 70)
    print("EVALUACIÓN EN IMAGEN DE PRUEBA (TEST)")
    print("=" * 70)
    test_oa, test_f1, test_kappa, cm = evaluate(model, test_loader, endmembers_tensor, device)

    print(f"Test Overall Accuracy (OA): {test_oa * 100:.2f}%")
    print(f"Macro F1-Score:             {test_f1:.4f}")
    print(f"Coeficiente Kappa:         {test_kappa:.4f}")
    print("\nMatriz de Confusión (Test):")
    print(cm)

    return test_oa, test_f1, test_kappa, cm