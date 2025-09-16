# E2SRLF: End-to-End Light Field Image Super-Resolution Depth Estimation


## Requirement

- PyTorch 1.13.0, torchvision 0.15.0. The code is tested with python=3.8, cuda=11.0.
- A GPU with enough memory

## Datasets

- We used the HCI 4D LF benchmark for training and evaluation. Please refer to the [benchmark website](https://lightfield-analysis.uni-konstanz.de/) for details.
- 

### Path structure

```
.
├── dataset
│   ├── training
│   └── validation
├── Figure
│   ├── paper_picture
│   └── hardware_picture
├── Hardware
│   ├── L3FNet
│   │   ├── bit_files
│   │   ├── hwh_files
│   │   └── project_code
│   ├── Net_prune
│   │   ├── bit_files
│   │   └── hwh_files
│   ├── Net_w2bit
│   │   ├── bit_files
│   │   └── hwh_files
│   └── Net_w8bit
│       ├── bit_files
│       └── hwh_files
├── implement
│   ├── L3FNet_implementation
│   └── data_preprocessing
├── jupyter
│   ├── network_execution_scripts
│   └── algorithm_implementation_scripts
├── model
│   ├── network_functions
│   └── regular_functions
├── param
│   └── checkpoints
└── Results
    ├── our_network
    │   ├── Net_Full
    │   └── Net_Quant
    ├── Necessity_analysis
    │   ├── Net_3D
    │   ├── Net_99
    │   └── Net_Undpp
    └── Performance_improvement_analysis
        ├── Net_Unprune
        ├── Net_8bit
        ├── Net_w2bit
        ├── Net_w8bit
        └── Net_prune
```

### Train

- Set the hyper-parameters in parse_args() if needed. We have provided our default settings in the realeased codes.
- You can train the network by calling implement.py and giving the mode attribute to train.  
    ``` python implement/implement.py --net Net_Full  --n_epochs 3000 --mode train --device cuda:1 ```

- Checkpoint will be saved to ./param/'NetName'.
  
### Valition and Test

- After loading the weight file used by your domain, you can call implement.py and giving the mode attribute to valid or test.
- The result files (i.e., scene_name.pfm) will be saved to ./Results/'NetName'.

### Results

#### Contrast with the state-of-the-art work



