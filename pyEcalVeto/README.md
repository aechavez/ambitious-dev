# Pure python implementation of BDT analysis

## Purpose: Faster development and eliminates the need for `ldmx-analysis` and other dependencies
## Requirements: Working install of `ldmx-sw-v2.3.0` or greater and `v12` samples
## Only tested with a container including `numpy`, `xgboost`, and `matplotlib` packages

Currently set to run the full segmentation + MIP tracking BDT

Example treeMaker command to make flat trees from event samples:
```
ldmx python3 treeMaker.py -i <absolute_path_to_inputs> -g <labels_for_input_eg_PN> --out <absolute_outdirs> -m <max_events>
```
There are other options for this command. More information can be found in `mods/ROOTmanager.py`

Example bdtMaker command to train a BDT model:
```
ldmx python3 bdtMaker.py -s <path_to_combined_signal_training_file> -b <path_to_bkg_file>
```
You will get a warning from XGBoost, but everything should work. The BDT takes some time to train, so I suggest training and evaluating on ~100 event background and signal samples first just to see how it works

Example bdtEval command to evaluate a trained model on testing samples:
```
ldmx python3 bdtEval.py -i <absolute_path_to_testing> -g <labels> --out <absolute_path_output_file_name>
```
