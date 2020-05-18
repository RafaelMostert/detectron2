# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from operator import itemgetter
import contextlib
import io
import numpy as np
import torch
import itertools
import json
import logging
import os
import tempfile
from collections import OrderedDict
from fvcore.common.file_io import PathManager
from PIL import Image
from tabulate import tabulate

from detectron2.data import MetadataCatalog
from detectron2.utils import comm

from .evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)


class LOFAREvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, 
    outputs using LOFAR relevant metrics.
    The relevant metric measures whether a proposed detection box for the central source is able to
    capture all and only the sources associated to a single source as determined by crowdsourced
    associations in LGZ.
    That is: for all proposed boxes that cover the middle pixel of the input image check which
    sources from the component catalogue are inside. 
    The predicted box can fail in three different ways:
    1. No predicted box covers the middle pixel
    2. The predicted box misses a number of components
    3. The predicted box encompasses too many components
    4. The prediction score for the predicted box is lower than other boxes that cover the middle
        pixel
    5. The prediction score is lower than x
    """

    def __init__(self, dataset_name, output_dir, distributed=True):
        """
        Args:
            dataset_name (str): name of the dataset
            output_dir (str): output directory to save results for evaluation
        """
        #self._metadata = MetadataCatalog.get(dataset_name)
        """
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        """
        self._output_dir = output_dir
        self._dataset_name = dataset_name

        self._distributed = distributed
        self._cpu_device = torch.device("cpu")
        self._predictions_json = os.path.join(output_dir, "predictions.json")

    def reset(self):
        self._predictions = []
        self.focussed_comps = []
        self.related_comps = []
        self.unrelated_comps = []
        self.n_comps = []
        self.pred_bboxes_scores = []

    def process(self, inputs, outputs):
        # Save ground truths and predicted bounding boxes to this class
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"], "file_name":input["file_name"],
                    "focussed_comp":input["focussed_comp"],"related_comp":input["related_comp"],
                    "unrelated_comp":input["unrelated_comp"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)#.numpy()
                prediction["instances"] = instances
            self._predictions.append(prediction)

        self.focussed_comps = [[p["focussed_comp"][0][0], p["focussed_comp"][1][0]] 
                for p in self._predictions]
        self.related_comps = [p["related_comp"] if len(p["related_comp"])>0 else [[],[]] 
                for p in self._predictions]
        self.unrelated_comps = [p["unrelated_comp"] if len(p["unrelated_comp"])>0 else [[],[]] 
                for p in self._predictions]
        self.n_comps = [1+len(c[0]) if len(c[0])>0 else 1 for c in self.related_comps]
        
        # Get predicted bounding boxes per image as numpy arrays
        self.pred_bboxes_scores = [(image_dict['instances'].get_fields()['pred_boxes'].tensor.numpy(), 
                  image_dict['instances'].get_fields()['scores'].numpy()) 
                 for image_dict in self._predictions]

    def evaluate(self):
        # for parallel execution 
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            #self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        # Return empty if inputs and outputs are non-existing
        if len(self._predictions) == 0:
            logger.warning("[LOFAREvaluator] Did not receive valid predictions.")
            return {}

        # Save predicted instances
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)



        includes_associated_fail_fraction, includes_unassociated_fail_fraction = \
            self._evaluate_predictions_on_lofar_score()

        # Calculate/print catalogue improvement
        base_score = self.baseline()
        correct_cat = self.our_score(includes_associated_fail_fraction, includes_unassociated_fail_fraction)
        self.improv(base_score, correct_cat)

        self._results = OrderedDict()
        self._results["bbox"] = {"assoc_single_fail_fraction": includes_associated_fail_fraction[0],
        "assoc_multi_fail_fraction": includes_associated_fail_fraction[1],
        "unassoc_single_fail_fraction": includes_unassociated_fail_fraction[0],
        "unassoc_multi_fail_fraction": includes_unassociated_fail_fraction[1],
        "correct_catalogue": correct_cat}
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def baseline(self):
        total = self.single_comps + self.multi_comps
        correct = self.single_comps/total
        print(f"Baseline assumption cat is {correct:.1%} correct")
        return correct

    def our_score(self,assoc_fail, unassoc_fail, suffix=''):
        fail_single = assoc_fail[0]*self.single_comps + unassoc_fail[0]*self.single_comps
        fail_multi = assoc_fail[1]*self.multi_comps + unassoc_fail[1]*self.multi_comps
        total = self.single_comps + self.multi_comps
        correct = (total-(fail_single+fail_multi))/total
        print(f"Trained cat is {correct:.1%} correct")
        return correct

    def improv(self, baseline, our_score):
        print(f"{(our_score-baseline)/baseline:.2%} improvement")

    def is_within(self, x,y,xmin,ymin,xmax,ymax):
        """Return true if x, y lies within xmin,ymin,xmax,ymax.
        False otherwise.
        """
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
        else:
            return False

    def _evaluate_predictions_on_lofar_score(self, scale_factor=1, debug=False, imsize=200):
        """ 
        Evaluate the results using our LOFAR appropriate score.

            Evaluate self._predictions on the given tasks.
            Fill self._results with the metrics of the tasks.

            That is: for all proposed boxes that cover the middle pixel of the input image check which
            sources from the component catalogue are inside. 
            The predicted box can fail in three different ways:
            1. No predicted box covers the focussed box
            2. The predicted central box misses a number of components
            3. The predicted central box encompasses too many components
            4. The prediction score for the predicted box is lower than other boxes that cover the middle
                pixel
            5. The prediction score is lower than x
        
        """
        if debug:
            #Check ground truth and prediction values of first item
            print("scale_factor", scale_factor)
            print("focus, related, unrelated, ncomp")
            print(self.focussed_comps[0], self.related_comps[0], self.unrelated_comps[0], self.n_comps[0])
            #print(np.shape(self.focussed_comps), np.shape(self.related_comps),
            #        np.shape(self.unrelated_comps), np.shape(self.n_comps))
            print("ncomp",self.n_comps)
            print("pred_bboxes_scores")
            print(self.pred_bboxes_scores[0])

        # Count number of components in dataset
        # Retrieve number of components per central source
        #comps = [counts[comp_name_to_source_name_dict[source_name]] for source_name in source_names]
        
        # Get number of single and multi comp sources
        self.single_comps = sum([1 if n==1 else 0 for n in self.n_comps])
        self.multi_comps = sum([1 if n>1 else 0 for n in self.n_comps])
        print(f"We have {self.single_comps} single comp cutouts and {self.multi_comps} multi")

        # Filter out predicted bboxes that do not cover the focussed pixel
        pred_central_bboxes_scores = [[(tuple(bbox),score) for bbox, score in zip(bboxes, scores) 
                            if self.is_within(x*scale_factor,imsize-y*scale_factor, 
                                bbox[0],bbox[1],bbox[2],bbox[3])] 
                              for (x, y), (bboxes, scores) 
                              in zip(self.focussed_comps, self.pred_bboxes_scores)]
        if debug:
            print("pred_bboxes_scores after filtering out the focussed pixel")
            print(pred_central_bboxes_scores[0])
        
        # Take only the highest scoring bbox from this list of bboxes
        pred_central_bboxes_scores = [sorted(bboxes_scores, key=itemgetter(1), reverse=True)[0] 
                                      if len(bboxes_scores) > 0 else [[-1,-1,-1,-1],0] 
                                      for bboxes_scores in pred_central_bboxes_scores]
        if debug:
            print("pred_bboxes_scores after getting the highest scoring bbox")
            print(pred_central_bboxes_scores[0])

        # Check if other source comps fall inside predicted central box
        # NOTE: y is flipped here in a hacky way
        #print("comps")
        #[print(comps) for comps, (bbox, score) 
        #                                  in zip(self.related_comps, pred_central_bboxes_scores)]
        self.comp_scores = [np.sum([self.is_within(x*scale_factor,imsize-y*scale_factor,
            bbox[0],bbox[1],bbox[2],bbox[3]) 
                        for x,y in list(zip(comps[0],comps[1]))])
                                          for comps, (bbox, score) 
                                          in zip(self.related_comps, pred_central_bboxes_scores)]
        if debug:
            print("comp_scores")
            print(self.comp_scores[0])

        # 1&2. "Predicted central bbox not existing or misses a number of components" can now be checked
        includes_associated_fail_fraction = self._check_if_pred_central_bbox_misses_comp(debug=debug)
        
        # 3&4. "Predicted central bbox encompasses too many or too few components" can now be checked
        if debug:
            print('len comp_scores ',len(self.comp_scores))
        assert len(self.unrelated_comps) == len(pred_central_bboxes_scores)
        # NOTE: y is flipped here in a hacky way
        self.close_comp_scores = [np.sum([self.is_within(x*scale_factor,imsize-y*scale_factor,
            bbox[0],bbox[1],bbox[2],bbox[3]) 
                    for x,y in zip(xs,ys)])
                            for (xs,ys), (bbox, score) in zip(self.unrelated_comps,
                                pred_central_bboxes_scores)]
        includes_unassociated_fail_fraction = \
            self._check_if_pred_central_bbox_includes_unassociated_comps(debug=debug)


        return includes_associated_fail_fraction, includes_unassociated_fail_fraction

        
    def _check_if_pred_central_bbox_includes_unassociated_comps(self, debug=False):
        """Check whether the predicted central box includes a number of unassocatiated components
            as indicated by the ground truth"""
        # Tally for single comp
        single_comp_success = [total == 0 for n_comp, total in zip(self.n_comps, self.close_comp_scores) 
                               if n_comp == 1]
        single_comp_success_frac = np.sum(single_comp_success)/len(single_comp_success)
            
        # Tally for multi comp
        multi_comp_binary_success = [total == 0 for n_comp, total in 
                                     zip(self.n_comps, self.close_comp_scores) if n_comp > 1]
        multi_comp_success = [total for n_comp, total in zip(self.n_comps, self.close_comp_scores) 
                                    if n_comp > 1]
        multi_comp_binary_success_frac = np.sum(multi_comp_binary_success)/len(multi_comp_binary_success)
        
        if debug:
            # Collect single comp sources that includ unassociated comps
            ran = list(range(len(self.close_comp_scores)))
            fail_indices = [i for i, n_comp, total in zip(ran, self.n_comps, self.close_comp_scores) 
                    if ((n_comp == 1) and (0 != total)) ]
            #collect_misboxed(pred, image_dir, output_dir, "unassoc_single_fail_fraction", fail_indices,
            #        source_names,metadata,gt_data,gt_locs)

            # Collect single comp sources that fail to include their gt comp
            fail_indices = [i for i, n_comp, total in zip(ran, self.n_comps, self.close_comp_scores) 
                    if ((n_comp > 1) and (0 != total)) ]
            #collect_misboxed(pred, image_dir, output_dir, "unassoc_multi_fail_fraction", fail_indices,
            #        source_names,metadata,gt_data,gt_locs)
        return 1-single_comp_success_frac, 1-multi_comp_binary_success_frac



    def _check_if_pred_central_bbox_misses_comp(self, debug=False):
        """Check whether the predicted central box misses a number of assocatiated components
            as indicated by the ground truth"""

        # Tally for single comp
        single_comp_success = [n_comp == (total+1) for n_comp, total in zip(self.n_comps, self.comp_scores) if n_comp == 1]
        single_comp_success_frac = np.sum(single_comp_success)/len(single_comp_success)
            
        # Tally for multi comp
        multi_comp_binary_success = [n_comp == (total+1) for n_comp, total in 
                                     zip(self.n_comps, self.comp_scores) if n_comp > 1]
        multi_comp_binary_success_frac = np.sum(multi_comp_binary_success)/len(multi_comp_binary_success)
        
        if debug:
            # Collect single comp sources that fail to include their gt comp
            ran = list(range(len(self.comp_scores)))
            fail_indices = [i for i, n_comp, total in zip(ran, self.n_comps, self.comp_scores) 
                    if ((n_comp == 1) and (n_comp != total)) ]
            #collect_misboxed(pred, image_dir, output_dir, "assoc_single_fail_fraction", fail_indices,
            #        source_names,metadata,gt_data,gt_locs, imsize)

            # Collect single comp sources that fail to include their gt comp
            fail_indices = [i for i, n_comp, total in zip(ran, self.n_comps, self.comp_scores) 
                    if ((n_comp > 1) and (n_comp != total)) ]
            #collect_misboxed(pred, image_dir,output_dir, "assoc_multi_fail_fraction", fail_indices,
            #        source_names,metadata,imsize)

        return 1-single_comp_success_frac, 1-multi_comp_binary_success_frac
    


if __name__ == "__main__":
    from detectron2.utils.logger import setup_logger

    logger = setup_logger()
    import argparse

    parser = argparse.ArgumentParser()
    #TODO
    parser.add_argument("--gt-json")
    parser.add_argument("--gt-dir")
    parser.add_argument("--pred-json")
    parser.add_argument("--pred-dir")
    args = parser.parse_args()

    from panopticapi.evaluation import pq_compute

    with contextlib.redirect_stdout(io.StringIO()):
        pq_res = pq_compute(
            args.gt_json, args.pred_json, gt_folder=args.gt_dir, pred_folder=args.pred_dir
        )
        _print_panoptic_results(pq_res)
