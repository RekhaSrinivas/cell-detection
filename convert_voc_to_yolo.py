import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

# Paths
root_dir = "BCCD"
annotations_dir = os.path.join(root_dir, "Annotations")
images_dir = os.path.join(root_dir, "JPEGImages")

# Classes
classes = ["RBC", "WBC", "Platelets"]

# Output dirs
output_images_dir = "data/images"
output_labels_dir = "data/labels"

for split in ["train", "val"]:
    os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)

# Read all image filenames
image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

def convert_annotation(xml_path, label_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_w = int(root.find("size/width").text)
    image_h = int(root.find("size/height").text)

    with open(label_path, "w") as f:
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")
            xmin = int(xmlbox.find("xmin").text)
            xmax = int(xmlbox.find("xmax").text)
            ymin = int(xmlbox.find("ymin").text)
            ymax = int(xmlbox.find("ymax").text)

            # Normalize to YOLO format
            x_center = (xmin + xmax) / 2.0 / image_w
            y_center = (ymin + ymax) / 2.0 / image_h
            width = (xmax - xmin) / image_w
            height = (ymax - ymin) / image_h

            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

# Process and copy files
for split, files in zip(["train", "val"], [train_files, val_files]):
    for fname in files:
        name, _ = os.path.splitext(fname)

        # Copy image
        src_img = os.path.join(images_dir, fname)
        dst_img = os.path.join(output_images_dir, split, fname)
        os.system(f'copy "{src_img}" "{dst_img}"')

        # Convert annotation
        src_ann = os.path.join(annotations_dir, name + ".xml")
        dst_lbl = os.path.join(output_labels_dir, split, name + ".txt")
        convert_annotation(src_ann, dst_lbl)

print("✅ Dataset converted and split into train/val")
