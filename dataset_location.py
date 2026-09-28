# specify the root location where u downloaded the dataset
root_location = "D:/Desktop/CMU/3D_learning/assignment2"

# Q3.3: True trains/evaluates on the extended 3-class (chair, car, plane) split.
# train_model.py / eval_model.py can also override this per run with --dataset.
use_full_dataset = False


def get_paths(full):
    """(SHAPENET_PATH, R2N2_PATH, SPLITS_PATH) for the chair-only or 3-class dataset."""
    dataset_name = "r2n2_shapenet_dataset_full" if full else "r2n2_shapenet_dataset"
    r2n2_path = f"{root_location}/{dataset_name}/r2n2"
    shapenet_path = f"{root_location}/{dataset_name}/shapenet"
    if full:
        splits_path = f"{root_location}/{dataset_name}/split_3c.json"  # split file contains data entry for 3 classes
    else:
        splits_path = f"{root_location}/{dataset_name}/split_03001627.json"  # split file contains data entry for 03001627 class
    return shapenet_path, r2n2_path, splits_path


SHAPENET_PATH, R2N2_PATH, SPLITS_PATH = get_paths(use_full_dataset)

CLASS_SYNSETS = {"chair": "03001627", "car": "02958343", "plane": "02691156"}
SYNSET_CLASSES = {v: k for k, v in CLASS_SYNSETS.items()}
