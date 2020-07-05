## Source codes for capsonte project Finding Network: Neural Architecture Search
Capstone project involved research of current state of Neural Architecture Search (NAS), followed by deeper study of GDAS
model based on [*Searching for A Robust Neural Architecture in Four GPU Hours*](https://arxiv.org/abs/1910.04465). 
### Usage
To search from scratch, create a folder to keep CIFAR10 and CIFAR100 data, move cifar10-split.txt and cifar100-split.txt
to the data folder path and pass --data_path folder's name
```python
python GDAS.py --save_dir ./save_dir/ --data_path ./data/ --batch_size 196 \
--paper_arch --dataset cifar10
```
To infer and fully train the searched architecture
```python
python basic-main.py --model_path ./save_dir/checkpoint/seed-1-basic.pth \
--save_dir ./full_train/ --data_path ./data/ --batch_size 64 --dataset cifar10
```
### Flags for experiments in capstone project (not available in official repo)
* --fix_reduction - fix reduction cell while training
* --mixed_prec - use mixed precision training (install NVIDIA [apex](https://github.com/NVIDIA/apex) first and uncomment
                                                import apex in GDAS.py)
* --paper_arch - use macro architecture described in paper as official repo's macro has 1 less normal cell in each block
* -- no_gumbel - dont use Gumbel-Max trick for sampling and do direct sampling by torch.multinomial 

Repo is based on official GDAS implementation https://github.com/D-X-Y/AutoDL-Projects