# DA5402 Assign 8
## Nikshay Jain | MM21B044

```bash
project-root/
├── .git/                     # Git repository data (use version control!)
├── .gitignore                # Specifies intentionally untracked files Git should ignore
├── MLproject                 # Defines the MLflow Project structure, dependencies, entry points [cite: 2, 10]
├── README.md                 # High-level project description, setup, and usage instructions
├── requirements.txt          # Project dependencies (or environment.yml for conda)
├── config/                   # Configuration files (e.g., parameters, paths, API settings)
│   └── config.yaml           # Example config file
├── data/                     # All project data
│   ├── raw/                  # The original, immutable data dump
│   └── processed/            # Cleaned, transformed data ready for modeling
│   └── test_images/          # Sample images for testing the API endpoint [cite: 8]
├── docs/                     # Detailed project documentation (optional but recommended)
├── logs/                     # Folder for custom application/script logs [cite: 6]
├── mlruns/                   # Default folder for MLflow tracking output (metrics, params, artifacts) [cite: 5]
├── notebooks/                # Jupyter notebooks for exploration, experimentation (optional)
│   └── data_exploration.ipynb
├── src/                      # Source code for your project [cite: 5]
│   ├── __init__.py           # Makes src a Python package
│   ├── data_processing.py    # Scripts for loading and transforming data
│   ├── train.py              # Script to train the model, including MLflow tracking [cite: 5, 6, 7]
│   ├── evaluate.py           # Script to evaluate the model
│   ├── predict.py            # Script for making predictions with a trained model [cite: 11]
│   ├── api.py                # Code for the REST API endpoint using the trained model [cite: 8]
│   └── utils.py              # Utility functions used across the project
├── tests/                    # Automated tests (unit, integration)
│   ├── __init__.py
│   └── test_train.py         # Example test file
└── models/
```