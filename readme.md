## 版本
pytorch 2.1.1

python 3.11.5

cuda 12.5

其他详见 requirements.txt

## 训练代码
```
python main.py -model vit_xception_center --run_type train --split split2 -ms 100 -init 2 -incre 1 -d 0

-model: 指定模型，在utils.factory中进行选择
--run_type: 训练为train, 测试为test
--split: split2对应论文的protocol2, split5对应论文的protocol1
--ms: 指定了replay set的总大小，论文设置为100。当设置为0时，无replay set
--init: 指定了第一个session下有多少个model, 在split2中应指定为2, 在split5中应指定为1
--incre: 指定了新增的每个session为多少个model, 在本论文中恒定为1
-d: 指定运行的gpu
```

## 测试代码
```
python main.py -model vit_xception_center --run_type test --split split2 -ms 100 -init 2 -incre 1 -d 0 --skip
需要加载模型权重进行测试时，修改configs中vit_xception_center.json中的trained_path
另外，若测试为split5, 需在test_class中添加 [0]
```