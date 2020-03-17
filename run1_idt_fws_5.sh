#block(name=nnv-ipca, threads=4, memory=10000, gpus=1, hours=30, subtasks=1)
export CUDA_VISIBLE_DEVICES=3
source /home/souri/.virtualenvs/nnv/bin/activate
for i in `seq 1 5`; do
    python train_idt_fws.py data reccv2020_idt_fws_5 1 --seed $i --feat-window-size 5 > reccv2020_idt_fws_5_rtrain1-$i.txt && python inference_idt_fws.py data reccv2020_idt_fws_5 1 --seed $i --feat-window-size 5 > reccv2020_idt_fws_5_rtest1-$i.txt &
done
wait
for i in `seq 1 5`; do
    echo $i
    python eval.py data reccv2020_idt_fws_5 1 --seed $i > reccv2020_idt_fws_5_eval1-$i.txt
done
