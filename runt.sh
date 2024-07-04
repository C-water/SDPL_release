#!/bin/bash
intiate_args="$1"
#这里的$1表示输入的第一个参数
python initiate.py ${intiate_args}
# 保存原始参数
original_args="$1 --test_dir=/home/chenquan/2023ACMworkshop/LPN-main/Dataset/University1652/test --views=2 --batchsize=4 --h=512 --w=512 --block=4 --resnet --dense_LPN --gpu_ids=0"

# 循环执行脚本，每次将参数 pad 更改为不同的值
for pad_value in 0  # 0, 20 ,40 ,60 ,80 ,100
do
    # 构造新的参数
    new_args="${original_args} --pad=${pad_value}"

    # 执行脚本
    echo "Executing with pad=${pad_value}"
    python test_new.py ${new_args}
done
#这样就可以实现每次只需输入不同的name，然后依次重复上述代码，不需要修改intiate_args="--name=max"，这样就很方便了


