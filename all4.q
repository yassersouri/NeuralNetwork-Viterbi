#block(name=nnv-a-4, threads=4, memory=10000, gpus=1, hours=20, subtasks=5)
source /home/souri/.virtualenvs/nnv/bin/activate
python train.py data rcvpr2020 4 --seed $SUBTASK_ID
python inference.py data rcvpr2020 4 --seed $SUBTASK_ID
python eval.py data rcvpr2020 4 --seed $SUBTASK_ID
