# For Visual Genome:
a) Download the the images of Visual Genome [Part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) and [Part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Unzip and place all images in a folder ```data/vg/images/```

b) Download the annotations of [VG (in COCO-format)](https://drive.google.com/file/d/1aGwEu392DiECGdvwaYr-LgqGLmWhn8yD/view?usp=sharing) and unzip it in the ```data/``` forder.


# For Open HOIC-DET:
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
qpic
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```