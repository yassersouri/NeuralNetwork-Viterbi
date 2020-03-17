#block(name=nnv-ipca, threads=4, memory=10000, gpus=1, hours=30, subtasks=1)
export CUDA_VISIBLE_DEVICES=1
source ~/.virtualenvs/nnv/bin/activate
for i in `seq 1 5`; do
    python train.py data_i3d reccv2020_i3d_fws_1 1 --seed $i --feat-window-size 1 > reccv2020_i3d_fws_1_rtrain1-$i.txt && python inference.py data_i3d reccv2020_i3d_fws_1 1 --seed $i --feat-window-size 1 > reccv2020_i3d_fws_1_rtest1-$i.txt &
done
wait
for i in `seq 1 5`; do
    echo $i
    python eval.py data reccv2020_i3d_fws_1 1 --seed $i > reccv2020_i3d_fws_1_eval1-$i.txt
done
