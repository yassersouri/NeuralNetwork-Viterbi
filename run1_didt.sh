#block(name=nnv-ipca, threads=4, memory=10000, gpus=1, hours=30, subtasks=1)
export CUDA_VISIBLE_DEVICES=1
source /home/souri/.virtualenvs/nnv/bin/activate
for i in `seq 1 5`; do
    python train_didt.py data reccv2020_didt 1 --seed $i > reccv2020_didt_rtrain1-$i.txt && python inference_didt.py data reccv2020_didt 1 --seed $i > reccv2020_didt_rtest1-$i.txt &
done
wait
for i in `seq 1 5`; do
    echo $i
    python eval.py data reccv2020_didt 1 --seed $i > reccv2020_didt_eval1-$i.txt
done
