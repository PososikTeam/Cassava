  
REM #!/usr/bin/env bash

REM # Train model ligt aug, First try,  size = 456, batch = 8


REM # Train 4 folds on this data
python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.7 -t2 2.0 -optim adam -v -m efficientnet-b5 -b 8 -e 300 -s 456 -f 0 -a kaggle_light -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.7 -t2 2.0 -optim adam -v -m efficientnet-b5 -b 8 -e 300 -s 456 -f 1 -a kaggle_light -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.7 -t2 2.0 -optim adam -v -m efficientnet-b5 -b 8 -e 300 -s 456 -f 2 -a kaggle_light -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.7 -t2 2.0 -optim adam -v -m efficientnet-b5 -b 8 -e 300 -s 456 -f 3 -a kaggle_light -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.7 -t2 2.0 -optim adam -v -m efficientnet-b5 -b 8 -e 300 -s 456 -f 4 -a kaggle_light -lr 3e-4 
   