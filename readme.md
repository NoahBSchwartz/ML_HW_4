# Designing Explainable Speech-Based Models of Depression
**CSCI 5622: Machine Learning - Homework 4**

This project applies machine learning techniques to estimate depression severity (PHQ-8 scores) using the **Extended Distress Analysis Interview Corpus (E-DAIC)**. The study explores the trade-offs between interpretable models (Decision Trees) and complex Deep Learning architectures (LSTMs, Transformers), utilizing both linguistic (transcript) and acoustic (audio) modalities.

## 👥 Team Members
*   Aidan Wegner 
*   Jayan Agarwal
*   Kyle Fisch
*   Noah Schwartz

---

## 📂 Project Structure

```text
.
├── ML_HW_4_CUDA.ipynb    # Main Jupyter Notebook containing the full pipeline
├── README.md             # Project documentation
├── requirements.txt      # List of dependencies (see below)
└── Data/                 # Dataset folder (Not included in repo, see Setup)
    ├── DepressionLabels.csv
    └── EDAIC Acoustics/
        ├── 300_utterance_agg.csv
        ├── 301_utterance_agg.csv
        └── ...
```

## 🛠️ Prerequisites & Installation

To run this notebook, you need **Python 3.8+** and the following libraries. 

**Note:** This project utilizes PyTorch with CUDA acceleration. While it will run on a CPU, a GPU is highly recommended for the Deep Learning and Transformer sections.

### 1. Install Dependencies
Run the following command to install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch transformers shap nltk
```

### 2. Data Setup
Due to the size and privacy constraints of the E-DAIC dataset, the data is not included in this repository. 
1.  Ensure you have the `DepressionLabels.csv` file.
2.  Ensure you have the `EDAIC Acoustics` folder containing the per-participant CSV files (eGeMAPS features).
3.  **Important:** Open the Jupyter Notebook (`ML_HW_4_CUDA.ipynb`) and modify **Cell 2** to point to your local data directory:

```python
# In Cell 2
BASE_PATH = "/path/to/your/Data/folder/" 
```

---

## 🚀 Methodology & Pipeline

The notebook is organized into the following sequential parts:

### Part (a): Feature Extraction
*   **Linguistic:** Extracts text from transcripts using **TF-IDF** (Syntactic) and **VADER** Sentiment Analysis (Semantic).
*   **Acoustic:** Aggregates time-series OpenSMILE features into statistical functionals (Mean, Std, Min, Max).

### Part (b): Interpretable Models (Text)
*   Trains **Decision Tree Regressors** on linguistic features.
*   Performs hyperparameter tuning on tree depth.
*   Visualizes the tree structure to interpret key decision nodes (e.g., "therapy", "medication").

### Part (c): Interpretable Models (Acoustic)
*   Trains Decision Trees on aggregated acoustic features.
*   Identifies top audio biomarkers (e.g., Shimmer, Spectral Flux).

### Part (d): Deep Learning Models
*   **Audio Model:** Dense Neural Network (MLP).
*   **Text Model:** LSTM (Long Short-Term Memory) network with word embeddings.
*   **Multimodal Model:** Late fusion architecture combining Audio and Text streams.
*   *Implemented using PyTorch with 5-Fold Cross-Validation.*

### Part (e): Explainability (SHAP)
*   Uses **SHAP (Shapley Additive exPlanations)** with a Kernel Explainer to interpret the Text LSTM.
*   Visualizes feature impact to show which specific words contribute to a high depression prediction.

### Part (f): Transformers (Bonus)
*   Utilizes **DistilBERT** (Hugging Face) to extract context-aware embeddings.
*   Trains a Ridge Regression model on top of BERT CLS tokens to compare against the LSTM.

---

## 📊 Results Summary

| Model Architecture | Average Pearson's $r$ | Relative Error |
| :--- | :--- | :--- |
| **Decision Tree (Text)** | 0.0218 | 0.2254 |
| **Decision Tree (Audio)** | 0.0367 | 0.2502 |
| **Deep Learning (Audio)** | 0.0896 | 0.1988 |
| **Deep Learning (Text LSTM)** | **0.1286** | **0.1878** |
| **Multimodal Fusion** | 0.1231 | 0.1891 |
| **Transformer (BERT)** | -0.0805 | 0.2081 |

*Note: The LSTM performed best, likely due to its ability to capture sequential context without the high-dimensional noise introduced by the Transformer on this small dataset.*

---

## 🖼️ Visualizations
The notebook generates the following plots for analysis:
1.  **PHQ-8 Distribution:** Histogram of depression scores in the dataset.
2.  **Decision Tree Plot:** Visualization of decision logic for text features.
3.  **Feature Importance:** Bar charts for top linguistic and acoustic features.
4.  **SHAP Summary Plot:** Visual explanation of LSTM decisions.
5.  **Predictions vs Actuals:** Scatter plot for model performance evaluation.

---

## 📜 Acknowledgments
*   Dataset: Extended Distress Analysis Interview Corpus (E-DAIC/AVEC 2019).
*   VADER Sentiment Analysis (Hutto & Gilbert, 2014).
*   SHAP (Lundberg & Lee, 2017).
```