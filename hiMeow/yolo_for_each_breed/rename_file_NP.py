import os

subfolder_mapping = {
    "무": "negative",
    "유": "positive"
}

base_path = "../../dataset"

def rename_subfolders(base_path, subfolder_mapping):
    for parent_folder in ["Training", "Validation"]:
        current_path = os.path.join(base_path, parent_folder)
        if not os.path.exists(current_path):
            print(f"Path not found: {current_path}")
            continue
        for disease_folder in os.listdir(current_path):
            disease_path = os.path.join(current_path, disease_folder)
            if not os.path.isdir(disease_path):
                continue
            for subfolder_name in os.listdir(disease_path):
                if subfolder_name in subfolder_mapping:
                    old_path = os.path.join(disease_path, subfolder_name)
                    new_path = os.path.join(disease_path, subfolder_mapping[subfolder_name])
                    try:
                        os.rename(old_path, new_path)
                        print(f'Renamed: {old_path} -> {new_path}')
                    except Exception as e:
                        print(f"Error renaming {old_path}: {e}")

rename_subfolders(base_path, subfolder_mapping)
