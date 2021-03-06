#!/usr/bin/env python

# # Import and load Detectron2 and libraries
#import torch, torchvision
print("Setup detectron2 logger")
from detectron2.utils.logger import setup_logger
setup_logger()
print("Import some common detectron2 utilities")
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, LOFARTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LOFAREvaluator

# import some common libraries
import numpy as np
from sys import argv
from cv2 import imread
import random
import os
import pickle
from operator import itemgetter
import matplotlib.pyplot as plt
random.seed(5455)

assert len(argv) > 1, "Insert path of configuration file when executing this script"
cfg = get_cfg()
cfg.merge_from_file(argv[1])
if len(argv) == 3:
    start_dir = argv[2]
    print("Beginning of paths:", start_dir)
    cfg.DATASET_PATH = cfg.DATASET_PATH.replace("/data/mostertrij",start_dir)
    cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace("/data/mostertrij",start_dir)
    cfg.DATASETS.IMAGE_DIR = cfg.DATASETS.IMAGE_DIR.replace("/data/mostertrij",start_dir)
print(f"Loaded configuration file {argv[1]}")
#ROTATION_ENABLED = bool(int(argv[2])) # 0 is False, 1 is True
DATASET_PATH= cfg.DATASET_PATH
print(f"Experiment: {cfg.EXPERIMENT_NAME}")
print(f"Rotation enabled: {cfg.INPUT.ROTATION_ENABLED}")
print(f"Precomputed bboxes: {cfg.MODEL.PROPOSAL_GENERATOR}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)


print("Load our data")
#def get_lofar_dicts(annotation_filepath, n_cutouts=np.inf, rotation=False):
def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    new_data = []
    counter=1
    max_value = np.inf
    if annotation_filepath.endswith('train.pkl'): 
        max_value = min(cfg.DATASETS.TRAIN_SIZE,len(dataset_dicts))
    for i in range(len(dataset_dicts)):
        if counter > max_value:
            break
        for ob in dataset_dicts[i]['annotations']:
            ob['bbox_mode'] = BoxMode.XYXY_ABS
        if cfg.MODEL.PROPOSAL_GENERATOR:
            dataset_dicts[i]["proposal_bbox_mode"] = BoxMode.XYXY_ABS
        if cfg.INPUT.ROTATION_ENABLED:
            if len(argv) == 3:
                dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
            new_data.append(dataset_dicts[i])
            counter+=1 
        else:
            if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
                if len(argv) == 3:
                    dataset_dicts[i]['file_name'] = dataset_dicts[i]['file_name'].replace("/data/mostertrij",start_dir)
                new_data.append(dataset_dicts[i])
                counter+=1
    print('len dataset is:', len(new_data), annotation_filepath)
    return new_data

# Register data inside detectron
# With DATASET_SIZES one can limit the size of these datasets
for d in ["train", "val", "test"]:
    DatasetCatalog.register(d, 
                            lambda d=d:
                            get_lofar_dicts(os.path.join(
                                DATASET_PATH,f"VIA_json_{d}.pkl")))
    MetadataCatalog.get(d).set(thing_classes=["radio_source"])
lofar_metadata = MetadataCatalog.get("train")


# # Train mode

# To implement the LOFAR relevant metrics I changed
# DefaultTrainer into LOFARTrainer
# where the latter calls LOFAREvaluator within build_hooks instead of the default evaluator
# this works for the after the fact test eval
# for train eval those things are somewhere within a model 
# specifically a model that takes data and retuns a dict of losses
print("Load model")
trainer = LOFARTrainer(cfg) 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
pretrained_model_path = os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
if os.path.exists(pretrained_model_path):
    cfg.MODEL.WEIGHTS = pretrained_model_path
    trainer.resume_or_load(resume=True)
else:
    trainer.resume_or_load(resume=False)

print("Start training")
trainer.train()
#print("Start evaluation on val")
#val_loader = build_detection_test_loader(cfg, f"val")
#evaluator = LOFAREvaluator(f"val", cfg.OUTPUT_DIR, distributed=True,debug=True)
#predictions = inference_on_dataset(trainer.model, val_loader, evaluator, overwrite=True)
print("Start evaluation on test")
test_loader = build_detection_test_loader(cfg, f"test")
evaluator = LOFAREvaluator(f"test", cfg.OUTPUT_DIR, distributed=True,debug=True)
predictions = inference_on_dataset(trainer.model, test_loader, evaluator, overwrite=False)


print('Done training.')
