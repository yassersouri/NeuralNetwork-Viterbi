#block(name=nnv-ipca, threads=4, memory=10000, gpus=1, hours=30, subtasks=1)
export CUDA_VISIBLE_DEVICES=3
source /home/souri/.virtualenvs/nnv/bin/activate
for i in `seq 1 5`; do
    python train_i3d_pca.py data reccv2020_i3d_pca 1 --seed $i > reccv2020_i3d_pca_rtrain1-$i.txt && python inference_i3d_pca.py data reccv2020_i3d_pca 1 --seed $i > reccv2020_i3d_pca_rtest1-$i.txt &
done
wait
for i in `seq 1 5`; do
    echo $i
    python eval.py data reccv2020_i3d_pca 1 --seed $i > reccv2020_i3d_pca_eval1-$i.txt
done
