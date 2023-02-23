
####################################
#### convnext
# SL1H
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 1 --save_path kvasir/bl/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 1 --save_path kvasir/bl/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 1 --save_path kvasir/bl/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 1 --save_path kvasir/bl/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 1 --save_path kvasir/bl/cnx_f4 --seed 4

# 2HML
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 2 --save_path kvasir/2hs/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 2 --save_path kvasir/2hs/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 2 --save_path kvasir/2hs/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 2 --save_path kvasir/2hs/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 2 --save_path kvasir/2hs/cnx_f4 --seed 4

# 4HML
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --n_heads 4 --save_path kvasir/4hs/cnx_f4 --seed 4

# 2HSL
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --balanced_pmh True --n_heads 2 --save_path kvasir/2hs_bal/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --balanced_pmh True --n_heads 2 --save_path kvasir/2hs_bal/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --balanced_pmh True --n_heads 2 --save_path kvasir/2hs_bal/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --balanced_pmh True --n_heads 2 --save_path kvasir/2hs_bal/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --balanced_pmh True --n_heads 2 --save_path kvasir/2hs_bal/cnx_f4 --seed 4

# LS
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_ls --hypar 0.05 --save_path kvasir/ls_005/cnx_f4 --seed 4

# MbLS
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_mbls --hypar 6 --save_path kvasir/mbls_6/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_mbls --hypar 6 --save_path kvasir/mbls_6/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_mbls --hypar 6 --save_path kvasir/mbls_6/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_mbls --hypar 6 --save_path kvasir/mbls_6/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_mbls --hypar 6 --save_path kvasir/mbls_6/cnx_f4 --seed 4

# mixup
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method mxp --hypar 0.2 --save_path kvasir/mxp_02/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method mxp --hypar 0.2 --save_path kvasir/mxp_02/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method mxp --hypar 0.2 --save_path kvasir/mxp_02/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method mxp --hypar 0.2 --save_path kvasir/mxp_02/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method mxp --hypar 0.2 --save_path kvasir/mxp_02/cnx_f4 --seed 4

# DCA
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_dca --save_path kvasir/dca/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_dca --save_path kvasir/dca/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_dca --save_path kvasir/dca/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_dca --save_path kvasir/dca/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_kvasir.csv --data_path data/kvasir/ --model convnext --method ce_dca --save_path kvasir/dca/cnx_f4 --seed 4
