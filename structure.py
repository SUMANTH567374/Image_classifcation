import os

# Folder structure relative to the current directory
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/data",
    "src/models",
    "src/evaluate",
    "src/utils",
    "app",
    "frontend",
    "scripts",
    "models",
    "tests",
    ".github/workflows"
]

# File structure with optional default content
files_with_content = {
    "notebooks/eda.ipynb": "",
    "notebooks/model_experiment.ipynb": "",

    "src/data/dataloader.py": "",
    "src/models/model.py": "",
    "src/models/train.py": "",
    "src/evaluate/metrics.py": "",
    "src/utils/helpers.py": "",
    "src/config.py": "",

    "app/main.py": "",
    "app/model_loader.py": "",
    "app/schemas.py": "",

    "frontend/app.py": "",

    "scripts/train.sh": "#!/bin/bash\npython run_training.py\n",
    "scripts/evaluate.sh": "#!/bin/bash\n# Add evaluation command here\n",

    "models/best_model.pth": None,  # Placeholder for binary model, skip writing

    "tests/test_data.py": "",
    "tests/test_api.py": "",

    ".github/workflows/ci.yml": "name: CI\non: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3",

    "Dockerfile": "",
    "docker-compose.yml": "",

    "requirements.txt": "torch\nfastapi\nuvicorn\npillow\ntorchvision\nscikit-learn\n",

    "dvc.yaml": "",
    "dvc.lock": "",

    ".env": "# Add environment variables here",
    ".gitignore": "__pycache__/\n*.pyc\n.env\nmodels/*.pth\n",

    "README.md": "# AI-MedScan: Pneumonia Detection from Chest X-rays\n\nThis is a deep learning pipeline for medical image classification.\n\n## Features\n- Modular code\n- FastAPI for deployment\n- DVC for data versioning\n"
}

# Step 1: Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Step 2: Create files with optional content
for file_path, content in files_with_content.items():
    if content is None:
        # For binary files like model.pth, just touch it
        open(file_path, 'a').close()
    else:
        with open(file_path, 'w') as f:
            f.write(content)

print("✅ Project folder structure with templates created successfully.")
