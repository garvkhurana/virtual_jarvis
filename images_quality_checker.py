import os, glob
from pathlib import Path

labels_dir = Path(r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\candice\Custom Workflow Object Detection.v2i.yolov11\labels")  
img_dir = Path(r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\candice\Custom Workflow Object Detection.v2i.yolov11\images")            

bad = []
out_of_range = []
zero_box = []
count_per_class = {}

for f in labels_dir.glob("*.txt"):
    lines = f.read_text().strip().splitlines()
    for L in lines:
        parts = L.split()
        if len(parts) < 5:
            bad.append((f.name, L))
            continue
        cid, x, y, w, h = parts[:5]
        try:
            cid_i = int(float(cid))
            count_per_class[cid_i] = count_per_class.get(cid_i, 0) + 1
            if float(w) <= 0 or float(h) <= 0:
                zero_box.append((f.name, L))
        except Exception:
            bad.append((f.name, L))

print("Bad label lines (format problems):", len(bad))
print("Zero-area boxes:", len(zero_box))
print("Class distribution (first 20):", sorted(count_per_class.items())[:20])
if bad:
    print("Example bad:", bad[:5])
if zero_box:
    print("Example zero-box:", zero_box[:5])
