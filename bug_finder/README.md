* *太懒了，所以直接在mentor的code的基础上改* *

# step 1

先测试一下基本的收敛曲线：

基本配置如下：

1. wmt 14 的数据集；
2. model超参：
    --batch_size=256 \
    --variable_update="replicated" \
    --independent_replica="true" \
    --use_synthetic_data="false" \
    --src_max_len=50 \
    --encoder_type="cudnn_lstm" \
    --direction="bi" \
    --learning_rate 0.01 \

全部参数见页尾

经过测试，训练时： 1. 经常性的train飞，2. 正常情况下loss也会处于139K左右，前2-3个batch后就停滞住，不正常！！(实际上对lr测试了从0.00001到0.001的情况，都有train飞的情况)


排除了数据读取进来的时候没有问题

================

查出主要原因是 ：use_synthetic_data 没关

稍微调试了一下以后，在单个数据上wer最好能够达到0.3，但是大数据集的情况下依然很糟糕，
虽然通过打印出 gradient/logits 等信息，能看得出来model确实在学习，但学习到的是简单的几种word
的拼接。这个暂时告一段落




有一个暂时解释不了的现象： 同样的一组参数，有时候能跑，有时候会崩，而且崩的时候报错均为：
```
Traceback (most recent call last):
  File "train.py", line 202, in <module>
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  File "/home/qizhe/Workspace/virenv_py3/lib/python3.5/site-packages/tensorflow/python/platform/app.py", line 126, in run
    _sys.exit(main(argv))
  File "train.py", line 195, in main
    train(model, config, hparams)
  File "train.py", line 159, in train
    batch_id, loss))
TypeError: a float is required
```
上述 bug fix 掉了： loss有的时候算出的结果为 NoneType， None； 所以保险一点print的时候不指定模式的类型直接print("Loss : ", loss)


```
Each model replica is totally independent.
num_gpus = 1, batch size = 256
all_reduce_spec : nccl
param_server_device : gpu
src_vocab_size : 50000
tgt_file_name : data/train.de
forget_bias : 1.0
embedding_dim : 512
gradient_repacking : 4
variable_consistency : strong
num_buckets : 5
src_max_len : 50
max_gradient_norm : 5.0
encoder_type : cudnn_lstm
learning_rate : 0.01
prefetch_data_to_device : False
variable_update : replicated
direction : bi
tgt_vocab_file : data/vocab.50K.de
num_encoder_layers : 4
optimizer : adam
use_synthetic_data : True
tgt_vocab_size : 50000
output_buffer_size : None
unk_id : 0
num_units : 512
unit_type : lstm
bos : <s>
agg_small_grads_max_bytes : 0
batch_size : 256
independent_replica : True
disable_shuffle : False
num_decoder_layers : 4
agg_small_grads_max_group : 10
src_file_name : data/train.en
dropout : 0.0
num_parallel_calls : 4
tgt_max_len : None
enable_profile : False
local_parameter_device : gpu
eos : </s>
num_keep_ckpts : 5
time_major : False
src_vocab_file : data/vocab.50K.en
```
