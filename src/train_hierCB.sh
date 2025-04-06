#!/bin/bash
##SBATCH -p sm
##SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2]
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=24000
#SBATCH --job-name="hiercb"
#SBATCH --output=../exp/log_%j.txt

set -x

lr=1e-3
p_depth=3
w_depth=2
u_depth=1
head=1
w_pco=1.0
pco_mg=1.0
w_clap=1.0
pco_ld=0.5
pco_lt=0.1
clap_t2a=0.5
w_p=1.0
w_w=1.0
w_u=1.0
ssl_drop=0.2
batch_size=25
embed_dim=24
model=hiercb
am=librispeech
seed_list=(22 33 44 55 66)

exp_dir=../exp/HierBFR_phnRes/3mV2_simple_pDep${p_depth}_wDep${w_depth}_uDep${u_depth}_p${w_p}_w${w_w}_u${w_u}_${w_pco}pco${pco_ld}ld${pco_lt}lt${pco_mg}mg_${w_clap}clap-${clap_t2a}t2a
# exp_dir=../exp/HierBFR_phnRes_conPco4_norm_tune/3mV2_simple_pDep${p_depth}_wDep${w_depth}_uDep${u_depth}_p${w_p}_w${w_w}_u${w_u}_${w_pco}pco${pco_ld}ld${pco_lt}lt${pco_mg}mg_${w_clap}clap-${clap_t2a}t2a


# repeat times
# repeat_list=(0 1 2 3 4)
repeat_list=(0)

for repeat in "${repeat_list[@]}"; do
    mkdir -pv $exp_dir/${repeat}
    python traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py \
        --lr ${lr} --exp-dir ${exp_dir}/${repeat} --p_depth ${p_depth} --hiercbheads ${head} \
        --batch_size ${batch_size} --embed_dim ${embed_dim} \
        --loss_w_phn ${w_p} --loss_w_word ${w_w} --loss_w_utt ${w_u} --loss_w_clap ${w_clap} --loss_w_pco ${w_pco} --pco_ld ${pco_ld} --pco_lt ${pco_lt} --pco_mg ${pco_mg} --clap_t2a ${clap_t2a} \
        --model ${model} --am ${am} --seed "${seed_list[$repeat]}" --ssl_drop ${ssl_drop} --w_depth ${w_depth} --u_depth ${u_depth}
    # python traintest_eng_dur_ssl_3m_HierBFR_conPCO_norm.py \
    #     --lr ${lr} --exp-dir ${exp_dir}/${repeat} --p_depth ${p_depth} --hiercbheads ${head} \
    #     --batch_size ${batch_size} --embed_dim ${embed_dim} \
    #     --loss_w_phn ${w_p} --loss_w_word ${w_w} --loss_w_utt ${w_u} --loss_w_clap ${w_clap} --loss_w_pco ${w_pco} --pco_ld ${pco_ld} --pco_lt ${pco_lt} --pco_mg ${pco_mg} --clap_t2a ${clap_t2a} \
    #     --model ${model} --am ${am} --seed "${seed_list[$repeat]}" --ssl_drop ${ssl_drop} --w_depth ${w_depth} --u_depth ${u_depth} --conpco
done
