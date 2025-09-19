"""
This script downloads a dataset from Kaggle using the kagglehub library and copies the downloaded files
to a specified target directory.

Steps performed:
1. Downloads the dataset "yousefsaeedian/ai-medical-chatbot" to kagglehub's cache directory.
2. Prints the path where the dataset was downloaded.
3. Ensures the target directory exists, creating it if necessary.
4. Copies all files and directories from the download path to the target directory.
5. Prints a confirmation message after copying is complete.

Dependencies:
- kagglehub
- shutil
- os

Attributes:
    target_dir (str): The directory where the dataset files will be copied.
"""
import kagglehub
import shutil
import os

# Download dataset (goes to kagglehub's cache)
path = kagglehub.dataset_download("yousefsaeedian/ai-medical-chatbot")
print("Downloaded to:", path)

# Your target directory (If you are running it locally the path might be different)
target_dir = r"/workspaces/medicortex.ai/data"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Move all files from download path to your target directory
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Files copied to:", target_dir)