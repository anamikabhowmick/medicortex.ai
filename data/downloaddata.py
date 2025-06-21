import kagglehub
import shutil
import os

# Download dataset (goes to kagglehub's cache)
path = kagglehub.dataset_download("yousefsaeedian/ai-medical-chatbot")
print("Downloaded to:", path)

# Your target directory
target_dir = r"D:\Data & Projects\medicortex.ai\data"

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