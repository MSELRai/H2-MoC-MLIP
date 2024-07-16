#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
for sd in 314 42 914;
	  do
	      nlayers=2
	      rc=5
	      lmax=1
	      body_order=2
	      nfeat=128
	      device=cuda  # cpu or cuda
	      nepoch=50
	      nswa=35
	      
	      mace_run_train \
		  --name="mace-$sd" \
		  --train_file="train.xyz" \
		  --valid_fraction=0.05 \
		  --test_file="test.xyz" \
		  --E0s="isolated" \
		  --energy_key="energy" \
		  --forces_key="forces" \
		  --stress_key="stress" \
		  --compute_stress=True \
		  --model="MACE" \
		  --num_interactions=$nlayers \
		  --num_channels=$nfeat \
		  --max_L=$lmax \
		  --correlation=$body_order \
		  --forces_weight=1000 \
		  --energy_weight=10 \
		  --r_max=$rc \
		  --batch_size=5 \
		  --valid_batch_size=5 \
		  --max_num_epochs=$nepoch \
		  --start_swa=$nswa \
		  --scheduler_patience=5 \
		  --patience=15 \
		  --eval_interval=1 \
		  --ema \
		  --swa \
		  --swa_forces_weight=10 \
		  --error_table="PerAtomRMSE" \
		  --default_dtype="float32" \
		  --device=$device \
		  --seed=$sd \
		  --restart_latest \
		  --save_cpu > /dev/null 2> /dev/null &

	      export CUDA_VISIBLE_DEVICES=$(($CUDA_VISIBLE_DEVICES + 1 ))
done
wait
