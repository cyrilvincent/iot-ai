from PIL import Image
import os

# Download url : https://www.microsoft.com/en-us/download/details.aspx?id=54765

def clean_directory(root_dir):
    removed = 0
    for subdir, _, files in os.walk(root_dir):
        for fname in files:
            path = os.path.join(subdir, fname)
            try:
                img = Image.open(path)
                img.verify()
            except Exception as e:
                print(f"❌ Supprimé : {path} ({e})")
                os.remove(path)
                removed += 1
    print(f"\n✅ Nettoyage terminé : {removed} images corrompues supprimées")

clean_directory(r'C:\cats_vs_dogs')