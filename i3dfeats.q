#block(name=nnv-i3d, threads=4, memory=10000, gpus=1, hours=96, subtasks=3)
source /home/souri/.virtualenvs/nnv/bin/activate
python train.py data_i3d results_i3d --seed $SUBTASK_ID
python inference.py data_i3d results_i3d --seed $SUBTASK_ID
python eval.py data_i3d results_i3d --seed $SUBTASK_ID
