#!/bin/bash

cd ..

# custom config
DATA=/root/dataset # you may change your path to dataset here
TRAINER=DAMP

DATASET=office_home # name of the dataset
CFG=damp  # config file
TAU=0.6 # pseudo label threshold
U=1.0 # coefficient for loss_u
SEED=1

# NAME=ac
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}/DAMPallzhengjiaoloss
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Art --target-domains Clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=ap
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Art --target-domains Product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

NAME=ar
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
DIR=output/counterfactual
python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Art --target-domains Realworld --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=ca
# # DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# DIR=output/counterfactual
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Clipart --target-domains Art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=cp
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Clipart --target-domains Product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=cr
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Clipart --target-domains Realworld --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=pa
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Product --target-domains Art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=pc
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Product --target-domains Clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=pr
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Product --target-domains Realworld --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=ra
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Realworld --target-domains Art --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=rc
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Realworld --target-domains Clipart --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}

# NAME=rp
# DIR=output/${DATASET}/${TRAINER}/${CFG}/${T}_${TAU}_${U}_${NAME}/seed_${SEED}
# python train.py --root ${DATA} --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml --config-file configs/trainers/${TRAINER}/${CFG}.yaml --output-dir ${DIR} --source-domains Realworld --target-domains Product --seed ${SEED}  TRAINER.DAMP.TAU ${TAU} TRAINER.DAMP.U ${U}