G1=10
G2=11
G3=12
G4=13
G5=14
G6=15

CUDA_VISIBLE_DEVICES=$G1 python train_lofar.py \
    configs/lofar_detection/iterations_v10_300kconstantLR.yaml &
CUDA_VISIBLE_DEVICES=$G2 python train_lofar.py \
    configs/lofar_detection/iterations_v11_300kconstantLR_withRot.yaml &
CUDA_VISIBLE_DEVICES=$G3 python train_lofar.py \
    configs/lofar_detection/iterations_v12_300kstepLR.yaml &
CUDA_VISIBLE_DEVICES=$G4 python train_lofar.py \
    configs/lofar_detection/iterations_v13_300kstepLR_withRot.yaml &
CUDA_VISIBLE_DEVICES=$G5 python train_lofar.py \
    configs/lofar_detection/iterations_v14_300kcosineLR.yaml &
CUDA_VISIBLE_DEVICES=$G6 python train_lofar.py \
    configs/lofar_detection/iterations_v15_300kcosineLR_withRot.yaml &
