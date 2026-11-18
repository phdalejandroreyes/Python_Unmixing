# Python Unmixing: Deep Learning & Spectral Unmixing for Hyperspectral Images

Proyecto de investigación y desarrollo centrado en el desmezclado de imágenes hiperespectrales (*Hyperspectral Unmixing*). El pipeline combina técnicas tradicionales de extracción de componentes (*endmembers*) con arquitecturas avanzadas de Aprendizaje Profundo, evaluando y comparando Modelos de Espacio de Estados (**Mamba / SSM**) y diversas variantes de **Transformers**.

---

## 📁 Estructura del Proyecto

```text
Python_unmixing/
├── main.py                             # Script principal de ejecución del pipeline
├── config.py                           # Carga y gestión de parámetros (config.yaml)
├── dataset.py                          # Extracción de parches 32x32 y PyTorch Dataloaders
├── losses.py                           # Funciones de pérdida (Focal Loss, entre otras)
├── training.py                         # Bucle de entrenamiento, validación y métricas intermedias
│
├── Satelital/                          # Imágenes hiperespectrales descargables desdeel drive: ______
│   ├── images                          # Imágenes hiperespectrales
│   └── labels                          # Etiquetas semánticas
│
├── Models/                             # Arquitecturas de Aprendizaje Profundo (Comparativa de investigación)
│   ├── model_builder.py                # Builder central (create_model) para selección dinámica
│   ├── mamba_transformer.py            # Modelo principal basado en SSM Mamba
│   ├── abundance_transformer.py        # Arquitectura alternativa de abundancias
│   ├── cross_attention_transformer.py  # Arquitectura alternativa con atención cruzada
│   ├── residual_transformer.py         # Arquitectura alternativa residual
│   └── resunet.py                      # Arquitectura alternativa ResUNet
│
├── unmixing/                           # Algoritmos tradicionales y preprocesamiento
│   ├── ebeae.py                        # Desmezclado lineal ciego
│   ├── nebeae.py                       # Desmezclado no lineal ciego
│   ├── ebeae_s.py                        # Desmezclado lineal semisupervisado al 20 porciento
│   ├── generate_pseudo_gt.py           # Generación de referencia Pseudo-GT
│   └── endmember_extraction.py         # Inicializaciones (VCA, N-FINDR, SVMAX)
│
├── Evaluation/                         # Módulo de Diagnóstico y Métricas
│   └── compute_metrics.py              # Cálculo cuantitativo de métricas (RMSE, SAD, etc.) vs Pseudo-GT
│
├── experiments/                        # Espacio de pruebas, versiones experimentales y scripts de depuración
│
├── .gitignore                          # Archivos ignorados por Git (datasets pesados, pesos .pth, venv)
└── README.md                           # Documentación y guía del proyecto
