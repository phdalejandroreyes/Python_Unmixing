# ============================================================
# dataset.py
#
# Dataset para mapas de abundancias
#
# Entrada:
#   A_initial -> (n_endmembers, num_pixels)
#   labels    -> (num_pixels,)
#
# Salida:
#   X -> (n_endmembers, patch_size, patch_size)
#   y -> (patch_size, patch_size)
#
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# DATASET
# ============================================================

class AbundanceDataset(Dataset):

    def __init__(
        self,
        A_initial,
        labels,
        image_shape,
        patch_size=32
    ):

        self.patch_size = patch_size

        # ----------------------------------------------------
        # Dimensiones espaciales
        # ----------------------------------------------------

        self.height = image_shape[0]
        self.width = image_shape[1]

        # ----------------------------------------------------
        # Convertir A_initial a numpy
        # ----------------------------------------------------

        if isinstance(A_initial, torch.Tensor):
            A_initial = A_initial.detach().cpu().numpy()

        A_initial = np.asarray(A_initial)

        # ----------------------------------------------------
        # Convertir labels a numpy
        # ----------------------------------------------------

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        labels = np.asarray(labels)

        # ----------------------------------------------------
        # Comprobar A_initial
        #
        # Esperamos:
        #
        # (n_endmembers, num_pixels)
        # ----------------------------------------------------

        if A_initial.ndim != 2:

            raise ValueError(
                "A_initial debe tener 2 dimensiones. "
                f"Shape recibido: {A_initial.shape}"
            )

        n_endmembers, num_pixels = A_initial.shape

        if num_pixels != self.height * self.width:

            raise ValueError(
                "El número de píxeles de A_initial "
                "no coincide con image_shape."
            )

        self.n_endmembers = n_endmembers

        # ----------------------------------------------------
        # Comprobar labels
        # ----------------------------------------------------

        labels = labels.reshape(-1)

        if labels.size != self.height * self.width:

            raise ValueError(
                "El número de labels no coincide "
                "con image_shape."
            )

        # ----------------------------------------------------
        # Convertir:
        #
        # A:
        # (n_endmembers, píxeles)
        #
        # a:
        # (n_endmembers, alto, ancho)
        # ----------------------------------------------------

        self.A = A_initial.reshape(
            n_endmembers,
            self.height,
            self.width
        )

        # ----------------------------------------------------
        # Convertir labels:
        #
        # (píxeles,)
        #
        # a:
        # (alto, ancho)
        # ----------------------------------------------------

        self.labels = labels.reshape(
            self.height,
            self.width
        )

        # ----------------------------------------------------
        # Comprobar divisibilidad
        # ----------------------------------------------------

        if self.height % patch_size != 0:

            raise ValueError(
                f"El alto ({self.height}) no es divisible "
                f"por patch_size ({patch_size})."
            )

        if self.width % patch_size != 0:

            raise ValueError(
                f"El ancho ({self.width}) no es divisible "
                f"por patch_size ({patch_size})."
            )

        # ----------------------------------------------------
        # Crear coordenadas
        # ----------------------------------------------------

        self.patch_coordinates = []

        for row in range(
            0,
            self.height,
            patch_size
        ):

            for col in range(
                0,
                self.width,
                patch_size
            ):

                self.patch_coordinates.append(
                    (row, col)
                )

    # ========================================================
    # Número de parches
    # ========================================================

    def __len__(self):

        return len(self.patch_coordinates)

    # ========================================================
    # Obtener muestra
    # ========================================================

    def __getitem__(self, index):

        row, col = self.patch_coordinates[index]

        # ----------------------------------------------------
        # Abundancias
        #
        # (n_endmembers, 32, 32)
        # ----------------------------------------------------

        A_patch = self.A[
            :,
            row:row + self.patch_size,
            col:col + self.patch_size
        ]

        # ----------------------------------------------------
        # Labels
        #
        # (32, 32)
        # ----------------------------------------------------

        label_patch = self.labels[
            row:row + self.patch_size,
            col:col + self.patch_size
        ]

        # ----------------------------------------------------
        # Convertir a tensors
        # ----------------------------------------------------

        A_patch = torch.tensor(
            A_patch,
            dtype=torch.float32
        )

        label_patch = torch.tensor(
            label_patch,
            dtype=torch.long
        )

        return A_patch, label_patch


# ============================================================
# DATALOADER
# ============================================================

def create_dataloader(
    A_initial,
    labels,
    image_shape,
    patch_size=32,
    batch_size=4,
    shuffle=True
):

    dataset = AbundanceDataset(
        A_initial=A_initial,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return dataset, dataloader
