# Frag_VAE
This repository contains our implementation of ""

# Requirements
* Python(version >= 3.6)
* Pytorch(version >= 1.1.0)
* RDKit (version >= 2019.03)
* networkx (version >= 2.4 )
* numpy (version >= 1.18 )

We highly recommend you to use conda for package management.

# Vocab Extraction
Fragments and interchangeable fragments are extracted by the following commond. (replace data/zinc_train.txt with your own molecular dataset)
```python
python vocab_extract.py --moldata data/zinc_train_pair.txt --frag_file zinc_vocab_frag_rd2.txt --bond_file zinc_vocab_bond_rd2.txt --radius 2 --ncpu 8
```

# Molecule Processing
Molecules are converted into three-level hierarchical graphs using the fragment vocabulary from the first step.
```python
mkdir zinc_train
python hier_mol_preprocess.py --moldata data/zinc_train_pair.txt --frag zinc_vocab_frag_rd2.txt --bond zinc_vocab_bond_rd2.txt --data_folder zinc_train/ --radius 2 --ncpu 8
```
# Model Training
Train the model with KL regularization, use
```python
mkdir zinc_model
python train.py --train_folder zinc_train --frag zinc_vocab_frag_rd2.txt --bond zinc_vocab_bond_rd2.txt --model_folder zinc_model/  --epoch 10
```
# Molecule Generation
To generate molecules, use
```python
python mol_generation.py --frag zinc_vocab_frag_rd2.txt --bond zinc_vocab_bond_rd2.txt --model_folder zinc_model/model.0 --fragment_num 4  --num_decode 20 --mol_num 5 --batch_size 5 --beam_size 0 --radius 2 --mol_file gen.txt --fragment_smiles '*[c:1]1ccc(C)cc1C'
```
