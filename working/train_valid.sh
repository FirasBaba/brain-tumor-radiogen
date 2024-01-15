python3 create_folds --n_folds 5

python3 -m train --fold 0 --type T1wCE --model_name resnet10
python3 -m train --fold 1 --type T1wCE --model_name resnet10
python3 -m train --fold 2 --type T1wCE --model_name resnet10
python3 -m train --fold 3 --type T1wCE --model_name resnet10
python3 -m train --fold 4 --type T1wCE --model_name resnet10

python3 -m validation

python3 -m predict --type T1wCE --model_name resnet10