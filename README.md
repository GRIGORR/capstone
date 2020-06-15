## Source codes for capsonte project Finding Network: Neural Architecture Search
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