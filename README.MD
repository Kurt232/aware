# Context Aware learning

## TARGET
Problem with Traditional DA:
1. Treats domain shifts as statistical (not structural), ignoring causal factors.
2. Limited generalization due to reliance on data-driven methods alone.
3. Conditional models focus on capacity, not root causes of distribution shifts.

Our Solution: Context-Aware Framework

Explicitly models causal links between context (e.g., environment, user behavior) and data shifts. Using context as condition to characterize the domain shift.

**Key Benefits**:
- Efficient Learning: Reduces need for labeled data via structured causal relationships.
- Broad Applicability: Works in supervised and unsupervised settings.
- Real-World Robustness: Maintains performance across diverse, unseen domains (e.g., HAR in varying environments).
- Interpretable: Clear insights into how context influences predictions.

## RELATED WORKS

Key words: 
1. HAR
2. Domain adaptation
3. Context-aware HAR
4. Conditional Neural Networks

## INFO

### install

python version >= 3.10

`pip install -r requirements.txt`

### scripts

./scripts/*_wo.sh: without aware

./run_*.sh: batch run scripts

### important files

data/dataset.py: dataset

models/units.py: basic model 

aware_train.py: with/without aware pre-training

train.py: with/without aware training (finetuning)

pretrain.py: deprecated file

### important path

/data/wjdu/hal/data: prepared dataset

/data/wjdu/hal/mk_data.py: make stand dataset

/data/wjdu/hal/mk_sync_data.py: make sync dataset

/data/wjdu/raw/: preprocessed dataset