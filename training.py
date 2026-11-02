# ============================================================
# training.py
#
# Módulo de entrenamiento y evaluación para el Transformer
# ============================================================

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix


def train_epoch(model, dataloader, endmembers_tensor, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for patches, labels in dataloader:
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(patches, endmembers_tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * patches.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, endmembers_tensor, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for patches, labels in dataloader:
            patches = patches.to(device)
            logits = model(patches, endmembers_tensor)
            preds = torch.argmax(logits, dim=1)  # Shape: (B, 32, 32)

            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_targets.append(labels.numpy().reshape(-1))

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    oa = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    kappa = cohen_kappa_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    return oa, f1, kappa, cm
def run_training_pipeline(model, train_loader, test_loader, endmembers_tensor, epochs, lr, device, criterion=None):
    # Si no se le pasa un criterio personalizado, usa CrossEntropyLoss estándar
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    print("\n" + "=" * 70)
    print("INICIANDO ENTRENAMIENTO DEL TRANSFORMER")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, endmembers_tensor, criterion, optimizer, device)
        elapsed = time.time() - start_time
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Época [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.6f} - Tiempo: {elapsed:.2f}s")

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