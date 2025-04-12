#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=luko113g@mailbox.tu-dresden.de
#SBATCH --account=p_scads_llm_secrets

source $HOME/venv/scadsai/bin/activate
python "$HOME/lk_scadsai/sNeuron-TST/src/identify.py" -t 10000 -ap "$HOME/lk_scadsai/sNeuron-TST/data/Hate_Speech/preprocessed" -o "$HOME/lk_scadsai/sNeuron-TST/data/Hate_Speech/neurons"
