REM #!/usr/bin/env bash

REM # Train model ligt aug, First try,  size = 300, batch = 32


REM # Train 4 folds on this data
python train.py --seed 103 --use-current -v -m efficientnet-b3 -b 32 -e 150 -s 300 -f 0 -a light -lr 3e-4 

python train.py --seed 103 --use-current -v -m efficientnet-b3 -b 32 -e 150 -s 300 -f 1 -a light -lr 3e-4 

python train.py --seed 103 --use-current -v -m efficientnet-b3 -b 32 -e 150 -s 300 -f 2 -a light -lr 3e-4 

python train.py --seed 103 --use-current -v -m efficientnet-b3 -b 32 -e 150 -s 300 -f 3 -a light -lr 3e-4 