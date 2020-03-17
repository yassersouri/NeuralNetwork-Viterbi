#block(name=nnv-r, threads=4, memory=10000, gpus=1, hours=48, subtasks=5)
source /home/souri/.virtualenvs/nnv/bin/activate
python train.py data results9 --seed $SUBTASK_ID
python inference.py data results9 --seed $SUBTASK_ID
python eval.py data results9 --seed $SUBTASK_ID
