# 📂 Data Directory Guide

Due to GitHub's strict file size limits (100MB per file), this repository does not contain the raw data or the large intermediate CSV files generated during the feature extraction process. 

Instead, all datasets required to reproduce this research are hosted externally. Please download the relevant files using the links below and place them directly into this `data/` directory before running the scripts.

## 1. Raw Dataset (Zenodo)
The original, unparsed global Bluesky dataset used for this thesis is publicly available on Zenodo. If you intend to run the entire pipeline from scratch (starting from Phase 1 data preprocessing), you will need to download the raw compressed archives (e.g., `interactions.csv.gz`) from here:

🔗 **[Raw Bluesky Dataset (Zenodo)](https://zenodo.org/records/14669616)**

## 2. Processed Data & Extracted Features (Google Drive)
Running the brute-force network extraction on the raw Zenodo dataset requires significant computational time. To facilitate instant reproducibility, we have provided all intermediate files, cleaned data, extracted Social Neighbourhood (SN) graphs (e.g., weak and strong ties), and the final feature matrices. 

If you wish to skip the data processing phase and directly evaluate our XGBoost models (Phase 3), you can download all the necessary pre-processed files here:

🔗 **[Processed Data & Feature Matrices (Google Drive)](https://drive.google.com/drive/folders/1O09Wju9TY8LxuLVfnKNt7CVoJ-5NKZbH?usp=sharing)**

---

### 💡 Usage Instructions
1. Download the required `.csv` or `.csv.gz` files from the links above.
2. Place them directly in this `data/` folder (do not create sub-folders unless specified by the scripts).
3. The Python scripts and Jupyter Notebooks in the parent directories are configured to read data using relative paths (e.g., `./data/your_file.csv`).