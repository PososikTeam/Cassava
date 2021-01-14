  
REM #!/usr/bin/env bash

REM # Train model ligt aug, First try,  size = 456, batch = 8


REM # Train 4 folds on this data
python train.py --seed 322 --use-current -l tempared_log_loss -t1 0.2 -t2 4.0 -optim adam -v -m SEResNext50 -b 8 -e 500 -s 460 -f 0 -a medium -lr 3e-4 


   