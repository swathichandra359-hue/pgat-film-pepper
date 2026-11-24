"""Run this first: checks Python packages, the GPU and the data folders.

  python check_setup.py data/bell_pepper data/black_pepper
"""
import os, sys

ok = True
print("Python", sys.version.split()[0])
for pkg in ["torch", "torchvision", "timm", "sklearn", "skimage", "imagehash", "pandas", "statsmodels", "matplotlib", "tabulate"]:
    try:
        mod = __import__(pkg); print(f"  OK   {pkg} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"  MISSING {pkg}  -> pip install -r requirements.txt"); ok = False

try:
    import torch
    if torch.cuda.is_available():
        print(f"  OK   GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("  PROBLEM: PyTorch cannot see the GPU. Reinstall torch with CUDA:\n"
              "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"); ok = False
except Exception as e:
    print("  torch error:", e); ok = False

for d in sys.argv[1:]:
    if not os.path.isdir(d):
        print(f"  PROBLEM: folder not found: {d}"); ok = False; continue
    classes = sorted(c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c)))
    print(f"  {d}: {len(classes)} classes")
    for c in classes:
        n = sum(f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) for _, _, fs in os.walk(os.path.join(d, c)) for f in fs)
        print(f"     {c}: {n} images")
    if len(classes) < 2:
        print("  PROBLEM: need one sub-folder per class (at least 2)"); ok = False

print("\nREADY - next: python run_all.py --data <folder> --name <name> --quick" if ok else "\nFix the problems above first.")
