# Information Diffusion and Network Influence on Bluesky

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU_Accelerated-ee4c2c.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-success.svg)

This repository contains the complete code pipeline for my Master's thesis, which investigates information diffusion and repost behaviour on the decentralized social network **Bluesky**. By integrating **Message (M) features, User (U) features, and structural Social Neighbourhood (SN) graphs (Strong vs. Weak ties)**, this project builds highly interpretable XGBoost models to predict user interaction dynamics.

## 🗂️ Repository Structure

The project follows a strict chronological data pipeline. Scripts are numbered sequentially to guarantee exact reproducibility.

### `data/`
Contains instructions for downloading the raw dataset and pre-processed feature matrices. *(Note: Large data files are not tracked in this repository due to size limits. See `data/README.md` for download links).*

### `1_data_preprocessing/`
Scripts to parse massive data dumps, reconstruct the global weak/strong tie interaction graphs, and filter targets.
* `00_data_exploration_peeking.py` - Initial peek into raw `.tar.gz` and `.csv.gz` structures.
* `01_find_top_hashtags.py` - Parses hashtags to define thematic boundaries.
* `02_extract_target_posts.py` - Extracts candidate posts (M) using strict quotas.
* `03_find_reposters.py` - Maps positive interaction edges (U_S -> U_R).
* `04_find_SN_weak_ties.py` - Extracts top-K weak tie graphs.
* `04.5_summary_statistics_for_SN_strong_ties.py` - Generates baseline statistics for strong ties within a 30-day temporal window.
* `05_find_SN_strong_ties.py` - Extracts dynamic, time-dependent strong ties (Quotes, Reposts, Comments).

### `2_features_extraction/`
The core NLP and feature engineering pipeline.
* `pre06.py` - Downloads HuggingFace models for offline, local execution.
* `06_extract_M_features.py` - Extracts computational heuristics, sentiment, and deep learning metrics (Toxic-BERT, RoBERTa) for candidate posts.
* `07_generate_experiment_set.py` - Generates negative samples for the 1:1 and 1:5 experimental settings.
* `08_generate_user_pool.py` - Maps local social neighbourhoods for all target users.
* `09_generate_historical_posts.py` - Extracts the historical timelines for the user pool.
* `10_extract_historical_posts_dl_features.py` - GPU-accelerated NLP processing of historical user timelines.

### `3_experiments_and_models/`
Final data assembly and Machine Learning evaluation.
* `11_data_assembly_and_split.py` - The "M2 Engine" that dynamically fuses M, U, and SN features and creates strictly temporal In-Distribution (ID) and Out-of-Distribution (OOD) splits.
* `12_XGBoost_ablation_matrix.py` - Core experimental script. Runs comprehensive ablation matrices (18 configurations with M, 17 configurations without M) to evaluate the predictive power of different Social Neighbourhood definitions.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/thesis-bluesky-network-influence.git
   cd thesis-bluesky-network-influence
   ```

2. **Install dependencies:**
   It is highly recommended to use a virtual Python environment.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you plan to run the Deep Learning feature extraction, please ensure you have [PyTorch installed with CUDA support](https://pytorch.org/get-started/locally/) for optimal performance.*

3. **Download the Data:**
   Please refer to `data/README.md` to download the required datasets and place them in the `data/` directory.

## 🚀 How to Run the Pipeline

To replicate the study, execute the scripts in numerical order from the root directory to ensure relative paths resolve correctly. 

**Phase 1: Preprocessing & Network Graphing**
```bash
python 1_data_preprocessing/01_find_top_hashtags.py
python 1_data_preprocessing/02_extract_target_posts.py
# ... proceed sequentially
```

**Phase 2: Feature Extraction**
*Ensure `pre06.py` is run first while connected to the internet to cache the models. Subsequent scripts will run entirely offline.*
```bash
python 2_features_extraction/pre06.py
python 2_features_extraction/06_extract_M_features.py
# ... proceed sequentially
```

**Phase 3: Machine Learning Evaluation**
```bash
python 3_experiments_and_models/11_data_assembly_and_split.py
python 3_experiments_and_models/12_XGBoost_ablation_matrix.py
```
*The XGBoost script will output extensive evaluation metrics (AUC, F1, Precision, Recall) optimized via dynamic thresholding, saved automatically as an Excel report.*

---
**Author:** VZGP2  
**Thesis:** Beyond the User: A Comparative Analysis of Social Neighbourhood Information for Reposting Prediction on Bluesky
