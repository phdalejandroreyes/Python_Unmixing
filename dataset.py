import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class HyperspectralPatchDataset(Dataset):
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
        self.patches_A = []
        self.patches_labels = []

        # Normalizar entradas a listas para manejar 1 o N imágenes
        A_list = A_initial if isinstance(A_initial, list) else [A_initial]
        labels_list = labels if isinstance(labels, list) else [labels]
        shapes_list = image_shape if isinstance(image_shape, list) else [image_shape]

        if len(A_list) != len(labels_list) or len(A_list) != len(shapes_list):
            raise ValueError("Las listas de A_initial, labels e image_shape deben tener la misma longitud.")

        # Extracción e independización de parches por cada imagen
        for A_init, lbls, shape in zip(A_list, labels_list, shapes_list):
            height, width = shape[0], shape[1]

            if isinstance(A_init, torch.Tensor):
                A_init = A_init.detach().cpu().numpy()
            A_init = np.asarray(A_init)

            if isinstance(lbls, torch.Tensor):
                lbls = lbls.detach().cpu().numpy()
            lbls = np.asarray(lbls)

            if A_init.ndim != 2:
                raise ValueError(f"A_initial debe tener 2 dimensiones. Shape recibido: {A_init.shape}")

            n_endmembers, num_pixels = A_init.shape
            if num_pixels != height * width:
                raise ValueError("El número de píxeles de A_initial no coincide con image_shape.")

            lbls = lbls.reshape(-1)
            if lbls.size != height * width:
                raise ValueError("El número de labels no coincide con image_shape.")

            A_grid = A_init.reshape(n_endmembers, height, width)
            labels_grid = lbls.reshape(height, width)

            # Manejo flexible de bordes por imagen
            pad_h = (patch_size - (height % patch_size)) % patch_size
            pad_w = (patch_size - (width % patch_size)) % patch_size

            if pad_h > 0 or pad_w > 0:
                A_padded = np.pad(A_grid, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
                labels_padded = np.pad(labels_grid, ((0, pad_h), (0, pad_w)), mode='reflect')
                padded_height = height + pad_h
                padded_width = width + pad_w
            else:
                A_padded = A_grid
                labels_padded = labels_grid
                padded_height = height
                padded_width = width

            # Extraer y almacenar parches independientes
            for row in range(0, padded_height, patch_size):
                for col in range(0, padded_width, patch_size):
                    A_patch = A_padded[:, row:row + patch_size, col:col + patch_size]
                    label_patch = labels_padded[row:row + patch_size, col:col + patch_size]
                    
                    self.patches_A.append(A_patch)
                    self.patches_labels.append(label_patch)

    def __len__(self):
        return len(self.patches_A)

    def __getitem__(self, index):
        A_patch = self.patches_A[index]
        label_patch = self.patches_labels[index]

        A_tensor = torch.from_numpy(A_patch).float()
        label_tensor = torch.from_numpy(label_patch).long()

        if self.augment:
            # 1. Rotación aleatoria en múltiplos de 90°
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                A_tensor = torch.rot90(A_tensor, k, dims=[1, 2])
                label_tensor = torch.rot90(label_tensor, k, dims=[0, 1])

            # 2. Volteo horizontal
            if torch.rand(1).item() < 0.5:
                A_tensor = torch.flip(A_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[1])

            # 3. Volteo vertical
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
    # Uso de la clase con el nuevo nombre
    dataset = HyperspectralPatchDataset(
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