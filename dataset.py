import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AbundanceDataset(Dataset):
    def __init__(
        self,
        A_initial,
        labels,
        image_shape,
        patch_size=32,
        augment=False
    ):
        self.patch_size = patch_size
        self.augment = augment
        self.height, self.width = image_shape[0], image_shape[1]

        # Conversión a NumPy
        if isinstance(A_initial, torch.Tensor):
            A_initial = A_initial.detach().cpu().numpy()
        A_initial = np.asarray(A_initial)

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)

        # Validaciones de dimensiones
        if A_initial.ndim != 2:
            raise ValueError(f"A_initial debe tener 2 dimensiones. Shape recibido: {A_initial.shape}")

        n_endmembers, num_pixels = A_initial.shape
        if num_pixels != self.height * self.width:
            raise ValueError("El número de píxeles de A_initial no coincide con image_shape.")

        self.n_endmembers = n_endmembers
        labels = labels.reshape(-1)

        if labels.size != self.height * self.width:
            raise ValueError("El número de labels no coincide con image_shape.")

        # Reshape a estructuras 2D/3D
        A_grid = A_initial.reshape(n_endmembers, self.height, self.width)
        labels_grid = labels.reshape(self.height, self.width)

        # Manejo flexible de bordes si la imagen no es divisible por patch_size
        pad_h = (patch_size - (self.height % patch_size)) % patch_size
        pad_w = (patch_size - (self.width % patch_size)) % patch_size

        if pad_h > 0 or pad_w > 0:
            self.A = np.pad(A_grid, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            self.labels = np.pad(labels_grid, ((0, pad_h), (0, pad_w)), mode='reflect')
            self.padded_height = self.height + pad_h
            self.padded_width = self.width + pad_w
        else:
            self.A = A_grid
            self.labels = labels_grid
            self.padded_height = self.height
            self.padded_width = self.width

        # Mapeo de parches no solapados
        self.patch_coordinates = []
        for row in range(0, self.padded_height, patch_size):
            for col in range(0, self.padded_width, patch_size):
                self.patch_coordinates.append((row, col))

    def __len__(self):
        return len(self.patch_coordinates)

    def __getitem__(self, index):
        row, col = self.patch_coordinates[index]

        # Extraer parches (N_EST, 32, 32) y (32, 32)
        A_patch = self.A[:, row:row + self.patch_size, col:col + self.patch_size]
        label_patch = self.labels[row:row + self.patch_size, col:col + self.patch_size]

        # Conversión a tensores de PyTorch
        A_tensor = torch.from_numpy(A_patch).float()
        label_tensor = torch.from_numpy(label_patch).long()

        # Aumento de datos espacial (solo activo si self.augment=True)
        if self.augment:
            # 1. Rotación aleatoria en múltiplos de 90° (0°, 90°, 180°, 270°)
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                A_tensor = torch.rot90(A_tensor, k, dims=[1, 2])
                label_tensor = torch.rot90(label_tensor, k, dims=[0, 1])

            # 2. Volteo horizontal aleatorio (50% probabilidad)
            if torch.rand(1).item() < 0.5:
                A_tensor = torch.flip(A_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[1])

            # 3. Volteo vertical aleatorio (50% probabilidad)
            if torch.rand(1).item() < 0.5:
                A_tensor = torch.flip(A_tensor, dims=[1])
                label_tensor = torch.flip(label_tensor, dims=[0])

        return A_tensor, label_tensor


def create_dataloader(
    A_initial,
    labels,
    image_shape,
    patch_size=32,
    batch_size=4,
    shuffle=True,
    augment=False
):
    dataset = AbundanceDataset(
        A_initial=A_initial,
        labels=labels,
        image_shape=image_shape,
        patch_size=patch_size,
        augment=augment
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )

    return dataset, dataloader