  
REM #!/usr/bin/env bash

REM # Train model ligt aug, First try,  size = 456, batch = 8


REM # Train 4 folds on this data
python train.py --seed 322 --use-current -l tempared_log_loss -optim lookahead -v -m efficientnet-b6 -b 7 -e 300 -s 528 -f 0 -a medium -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -optim lookahead -v -m efficientnet-b6 -b 7 -e 300 -s 528 -f 1 -a medium -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -optim lookahead -v -m efficientnet-b6 -b 7 -e 300 -s 528 -f 2 -a medium -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -optim lookahead -v -m efficientnet-b6 -b 7 -e 300 -s 528 -f 3 -a medium -lr 3e-4 

python train.py --seed 322 --use-current -l tempared_log_loss -optim lookahead -v -m efficientnet-b6 -b 7 -e 300 -s 528 -f 4 -a medium -lr 3e-4 
   