import os
from pathlib import Path

def count_images_in_folder(folder_path):
    # List of image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        return 0
    
    # Count image files
    count = sum(1 for file in os.listdir(folder_path) 
               if file.lower().endswith(image_extensions))
    return count

def main():
    # 'positive' and 'negative' folder paths
    positive_folder = r"C:\Users\82103\Desktop\Blepharitis\positive"
    negative_folder = r"C:\Users\82103\Desktop\Blepharitis\negative"
    
    # Count images in each folder
    positive_count = count_images_in_folder(positive_folder)
    negative_count = count_images_in_folder(negative_folder)
    
    # Print results
    print(f"Number of images in 'positive' folder: {positive_count}")
    print(f"Number of images in 'negative' folder: {negative_count}")
    print(f"Total number of images: {positive_count + negative_count}")

if __name__ == "__main__":
    main()
