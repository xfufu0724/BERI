# Bidirectional Entity-Relation Interaction for Enhanced Scene Graph Generation

## For Inference
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt
```

## For Data

### Visual Genome

```
BERI
└───data
│   └───vg
│       │   rel.json
│       │   test.json
│       |   train.json
|       |   val.json
|       └─  images
 :       :
```

### HICO-DET

```
BERI
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

## Training

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py 

## Evaluation

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --eval True