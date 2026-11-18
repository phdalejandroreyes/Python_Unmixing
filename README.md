# Python Unmixing: Deep Learning & Spectral Unmixing for Hyperspectral Images

Proyecto de investigación y desarrollo centrado en el desmezclado de imágenes hiperespectrales (*Hyperspectral Unmixing*). El pipeline combina técnicas tradicionales de extracción de componentes (*endmembers*) con arquitecturas avanzadas de Aprendizaje Profundo, evaluando y comparando Modelos de Espacio de Estados (**Mamba / SSM**) y diversas variantes de **Transformers**.

---

## ⚙️ Orquestadores del Pipeline: Main vs. Pruebas Experimentales

El proyecto cuenta actualmente con tres scripts principales de ejecución dependiendo del objetivo de la investigación:

*   **1. La aplicación principal (`main.py`):** 
    Pensado para la experimentación modular de desmezcla hiperespectral. Carga de manera dinámica las configuraciones desde `config.yaml`, permitiéndo cambiar libremente entre distintos métodos de desmezclado (`ebeae`, `nebeae`, `ebeae_semisupervisado`) y posteriormente seleccionar entre múltiples arquitecturas de atención / transformers(`model_builder`)  .
*   **2. El Piloto de Consenso Global (`pilot_pglobal_abundance.py`):** 
    Enfoque especializado para la estrategia avanzada del proyecto. Construye un diccionario de endmembers global de consenso (`P_global`) a partir de las 7 escenas de entrenamiento, proyecta los datos sobre un símplex invariante usando la versión semisupervisada (`ebeae_s`), añade un 9.º canal con la norma radiométrica y ejecuta una evaluación multieescena robusta (entrenando y validando para arrojar métricas detalladas por escena y globales).
*   **3. Benchmark Multi-Arquitectura (main_multi.py): Script de evaluación masiva para comparar el rendimiento de las 5 arquitecturas del proyecto (resunet, abundance, residual, cross_attention, mamba). Procesa el flujo de entrenamiento secuencialmente, genera reportes detallados en JSON, matrices de confusión y consolida los resultados en una tabla CSV global.
    

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
