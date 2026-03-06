# Resume Screening

This repository contains code, notebooks, and models for automated resume screening using BERT-based sequence classification. The aim is to classify resumes into several predefined categories to support recruitment workflows and candidate analysis.

## Contents Overview

- **Model configuration and scripts**: BERT training scripts and configurations.
- **Notebooks for experiments, analysis, and visualizations**.
- **Preprocessing and data exploration scripts**.
- **Results, evaluation metrics, and utility scripts**.

## Notebook Guide

Navigate the `notebooks/` folder for structured experimentation and analysis. Here’s a summary of the main notebook series:

- **00_eda**: Exploratory Data Analysis (EDA) – an initial dive into the data.
- **10_baseline**: A baseline experiment; serves as a simple, non-comprehensive initial model.
- **20–29, 60–63**: Model training experiments, with each notebook exploring different training strategies, model variants, fairness interventions, and related methods.
- **30–31**: Data-centric processing – notebooks focused on data cleaning, clustering, and other data transformation steps.
- **40–42**: Deep-dive analyses of model fairness and interpretability, including fairness metrics and integrated gradients.
- **50**: Generation of plots and figures tailored for inclusion in scientific manuscripts.

## Directory Structure

- `notebooks/` – Jupyter notebooks as described above.
- `notebooks/models/` – Pretrained and fine-tuned model configurations.
- `data/` – (git-ignored) Place your datasets here.
- `runs/` – Training run outputs and logs (git-ignored).

## Getting Started

1. **Clone the repository:**
    ```
    git clone https://github.com/yourusername/resume-screening.git
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Add your data:**  
   Place your datasets inside the `data/` folder.

4. **Explore Notebooks:**  
   Start with `00_eda` for EDA, and use the numbered guide above to pursue modeling, analysis, or visualization.

## Model & Config

- **Model:** BERT for Sequence Classification (9 classes)
- See `notebooks/models/bert_9classes_final/config.json` for model details.

## Results

Key baseline (BERT + sqrt_rw) performance:
  - **Accuracy:** 0.609
  - **Macro F1:** 0.621
  - **TPR Gap (Worst Robust):** 0.329
  - **TPR Gap (Macro Robust):** 0.116

## Notes

- Certain folders and large files are excluded via `.gitignore`.
- The project supports research and experimentation in resume screening and fairness-aware modeling.

## License

Specify your license information here.

## Contact

Questions or collaboration proposals? Please open an issue or contact [your email or GitHub handle].