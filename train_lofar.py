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
from detectron2.engine import DefaultPredictor, DefaultTrainer, LOFARTrainer
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
print(f"Loaded configuration file {argv[1]}")
ROTATION_ENABLED = bool(int(argv[2])) # 0 is False, 1 is True
EXPERIMENT_NAME= cfg.EXPERIMENT_NAME
DATASET_PATH= cfg.DATASET_PATH
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Rotation enabled: {ROTATION_ENABLED}")
print(f"Output path: {cfg.OUTPUT_DIR}")
print(f"Attempt to load training data from: {DATASET_PATH}")
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)


print("Load our data")
def get_lofar_dicts(annotation_filepath):
    with open(annotation_filepath, "rb") as f:
        dataset_dicts = pickle.load(f)
    new_data = []
    for i in range(len(dataset_dicts)):
        for ob in dataset_dicts[i]['annotations']:
            ob['bbox_mode'] = BoxMode.XYXY_ABS
        if ROTATION_ENABLED:
            new_data.append(dataset_dicts[i])
        else:
            if dataset_dicts[i]['file_name'].endswith('_rotated0deg.png'):
                new_data.append(dataset_dicts[i])
    return new_data

# Register data inside detectron
for d in ["train", "val", "test"]:
    DatasetCatalog.register(d, 
                            lambda d=d: get_lofar_dicts(os.path.join(DATASET_PATH,f"VIA_json_{d}.pkl")))
    MetadataCatalog.get(d).set(thing_classes=["radio_source"])
lofar_metadata = MetadataCatalog.get("train")



print("Sample and plot input data as sanity check")
dataset_dicts = get_lofar_dicts(os.path.join(DATASET_PATH,"VIA_json_train.pkl"))
for i, d in enumerate(random.sample(dataset_dicts, 3)):
    img = imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=lofar_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    a= vis.get_image()[:, :, ::-1]
    plt.figure(figsize=(15,15))
    plt.imshow(a)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR,f"random_input_example_for_sanity_check_{i}"))
    plt.close()


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



"""
# Look at training curves in tensorboard:
get_ipython().run_line_magic('load_ext', 'tensorboard')
#%tensorboard --logdir output --host "0.0.0.0" --port 6006
get_ipython().run_line_magic('tensorboard', '--logdir output  --port 6006')
# In local command line input 
#ssh -X -N -f -L localhost:8890:localhost:6006 tritanium
# Then open localhost:8890 to see tensorboard
"""


print('Done training. Enter inference mode')
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)



print("Sample and plot predicted data as sanity check")
aap = get_lofar_dicts(os.path.join(DATASET_PATH,f"VIA_json_val.pkl"))
for d in random.sample(aap, 3):
    if not d["file_name"].endswith('_rotated0deg.png'):
        continue
    im = imread(d["file_name"])
    outputs = predictor(im)
    print(outputs["instances"])
    v = Visualizer(im[:, :, ::-1],
                   metadata=lofar_metadata, 
                   scale=1, 
                  instance_mode=ColorMode.IMAGE #_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(10,10))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(os.path.join(cfg.OUTPUT_DIR,f"random_prediction_example_for_sanity_check_{i}"))
    plt.close()


print("Evaluate performance for validation set")
# returns a torch DataLoader, that loads the given detection dataset, 
# with test-time transformation and batching.
# Val_loader produces inputs that can enter the model for inference, 
# the results of which can be evaluated by the evaluator
# The return value is that which is returned by evaluator.evaluate()
val_loader = build_detection_test_loader(cfg, f"val")
evaluator = LOFAREvaluator(f"val", cfg.OUTPUT_DIR, distributed=True)
predictions = inference_on_dataset(trainer.model, val_loader, evaluator, overwrite=True)


"""
print("Evaluate performance for test set")
# returns a torch DataLoader, that loads the given detection dataset, 
# with test-time transformation and batching.
# Val_loader produces inputs that can enter the model for inference, 
# the results of which can be evaluated by the evaluator
# The return value is that which is returned by evaluator.evaluate()
val_loader = build_detection_test_loader(cfg, "test")
evaluator = LOFAREvaluator("test", cfg.OUTPUT_DIR, distributed=True)
predictions = inference_on_dataset(trainer.model, val_loader, evaluator, overwrite=True)
"""
