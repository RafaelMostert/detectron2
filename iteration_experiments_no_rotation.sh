G1=13
G2=14
G3=15

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py \
    configs/lofar_detection/iterations_v10_100kconstantLR.yaml > v10.txt &
CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py \
    configs/lofar_detection/iterations_v12_100kstepLR.yaml > v12.txt &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py \
    configs/lofar_detection/iterations_v14_100kcosineLR.yaml v13.txt &
