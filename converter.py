# convert_to_png.py
from pathlib import Path
from PIL import Image

SUPPORTED_EXTS = {".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}

def convert_directory(folder: str = "C:\\Users\\Voivode\\Desktop\\crano\\assets\\images", delete_originals: bool = False):
    folder = Path(folder)
    if not folder.exists():
        print(f"❌ Directory not found: {folder}")
        return

    converted = 0
    for img_path in sorted(folder.iterdir()):
        if img_path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        out_path = img_path.with_suffix(".png")
        if out_path == img_path:  # already png
            continue
        try:
            with Image.open(img_path) as im:
                im.save(out_path, "PNG")
            converted += 1
            print(f"✅ {img_path.name} → {out_path.name}")
            if delete_originals and out_path.exists():
                img_path.unlink()
        except Exception as e:
            print(f"❌ Failed {img_path.name}: {e}")

    print(f"\n🎉 Converted {converted} images to PNG.")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "assets/images"
    convert_directory(target)