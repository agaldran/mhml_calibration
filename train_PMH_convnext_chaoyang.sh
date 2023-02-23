
####################################
#### convnext
# SL1H
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 1 --save_path chaoyang/sl1h/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 1 --save_path chaoyang/sl1h/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 1 --save_path chaoyang/sl1h/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 1 --save_path chaoyang/sl1h/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 1 --save_path chaoyang/sl1h/cnx_f4 --seed 4

# 2HSL
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --balanced_mh True --n_heads 2 --save_path chaoyang/2hsl/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --balanced_mh True --n_heads 2 --save_path chaoyang/2hsl/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --balanced_mh True --n_heads 2 --save_path chaoyang/2hsl/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --balanced_mh True --n_heads 2 --save_path chaoyang/2hsl/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --balanced_mh True --n_heads 2 --save_path chaoyang/2hsl/cnx_f4 --seed 4

# 2HML
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 2 --save_path chaoyang/2hml/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 2 --save_path chaoyang/2hml/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 2 --save_path chaoyang/2hml/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 2 --save_path chaoyang/2hml/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 2 --save_path chaoyang/2hml/cnx_f4 --seed 4

# 4HML
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 4 --save_path chaoyang/4hml/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 4 --save_path chaoyang/4hml/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 4 --save_path chaoyang/4hml/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 4 --save_path chaoyang/4hml/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_PMH.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --n_heads 4 --save_path chaoyang/4hml/cnx_f4 --seed 4

# LS
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_ls --hypar 0.05 --save_path chaoyang/ls_005/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_ls --hypar 0.05 --save_path chaoyang/ls_005/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_ls --hypar 0.05 --save_path chaoyang/ls_005/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_ls --hypar 0.05 --save_path chaoyang/ls_005/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_ls --hypar 0.05 --save_path chaoyang/ls_005/cnx_f4 --seed 4

# MbLS
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_mbls --hypar 6 --save_path chaoyang/mbls_6/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_mbls --hypar 6 --save_path chaoyang/mbls_6/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_mbls --hypar 6 --save_path chaoyang/mbls_6/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_mbls --hypar 6 --save_path chaoyang/mbls_6/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_mbls --hypar 6 --save_path chaoyang/mbls_6/cnx_f4 --seed 4

# mixup
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method mxp --hypar 0.2 --save_path chaoyang/mxp_02/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method mxp --hypar 0.2 --save_path chaoyang/mxp_02/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method mxp --hypar 0.2 --save_path chaoyang/mxp_02/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method mxp --hypar 0.2 --save_path chaoyang/mxp_02/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method mxp --hypar 0.2 --save_path chaoyang/mxp_02/cnx_f4 --seed 4

# DCA
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_dca --save_path chaoyang/dca/cnx_f0 --seed 0
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_dca --save_path chaoyang/dca/cnx_f1 --seed 1
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_dca --save_path chaoyang/dca/cnx_f2 --seed 2
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_dca --save_path chaoyang/dca/cnx_f3 --seed 3
CUBLAS_WORKSPACE_CONFIG=:16:8 python train_other.py --csv_train data/train_chaoyang.csv --data_path data/chaoyang/ --model convnext --method ce_dca --save_path chaoyang/dca/cnx_f4 --seed 4
