# Data Directory

This folder contains all data used in the recommendation system project.

## Structure

- `raw/`
  - Original MovieLens 100K dataset
  - Downloaded automatically by the script
  - Should not be modified

- `processed/`
  - Placeholder for processed or intermediate data
  - Not required for running the project
  - Can be used for caching if needed

---

## How data is handled

- Data is downloaded automatically using:
  `src/data/download.py`
- Preprocessing is done during runtime using:
  - `src/data/preprocessing.py`
- No dataset files are stored in the repository

---

## Notes

- This project uses the MovieLens 100K dataset
- All splits (train / validation / test) are created dynamically
- Temporal splitting is used to avoid data leakage