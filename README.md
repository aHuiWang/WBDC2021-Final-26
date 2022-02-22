## **致谢**
本解决方案模型文件主要参考 [**RecBole**](https://github.com/RUCAIBox/RecBole)

## **运行环境**
+ python==3.6
+ torch==1.3.0
+ scikit-learn
+ tqdm
+ pandas
+ numpy
+ fasttext
+ numba

此处已经将环境打包,可以直接执行以下命令进入虚拟环境
```
source activate /home/tione/notebook/envs/wbdc
```
如需别处新建, 可以执行init.sh会在当前目录下生成一个新的虚拟环境，

## **目录结构**
/home/tione/notebook/wbdc2021-semi的目录结构如下所示：
```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements 
├── train.sh, script for preparing train/inference data and training models, including pretrained models 
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/test dataset
|   ├── train, codes for training
|   ├── inference.py, main function for inference on test dataset
|   ├── evaluation.py, main function for evaluation 
│   ├── model, codes for model architecture
│   ├── utils, all other codes which do not belong to previous directories
├── data
│   ├── submission, prediction result after running inference.sh
│   ├── model, model files (e.g. tensorflow checkpoints) 
```

## **运行流程**
### 训练

```shell script
sh train.sh
```

1.会进行数据预处理，包括数据采样，及得到预训练embedding各个序列文件

2.使用fasttext预训练对应的embedding

3.训练所需的若干模型，分别对应不同的超参设置，

### 融合预测
需要给出B榜数据路径
```shell script
sh inference.sh ./data/wedata/wechat_algo_data/test_a.csv
```
生成结果文件至./data/submission/result.csv

## **模型及特征**
主要使用基于DCN+MMOE和PNN+MMOE的模型融合
id类特征做了embedding,同时序列类特征使用GRU编码

使用的静态特征主要有 userid, feedid, authorid, bgm_song_id, bgm_singer_id, description_char, 
manual_tag_list, device, videoplayseconds

+ feedid, authorid, bgm_song_id, bgm_singer_id 根据用户观看日志组成对应的id序列，使用fasttext预训练得到初始化embedding
+ description_char使用所有视频 description_char, 
ocr_char, asr_char 对应的词id序列，使用fasttext预训练得到初始化embedding
+ userid 使用deepwalk在 user-feed图上随机游走得到序列，使用fasttext预训练得到初始化embedding

将用户日志中发生任意一种行为的feedid抽出构成历史序列,认为是用户动态变化的偏好

## **算法性能**
总预测时长: 646 s
单个目标行为2000条样本的平均预测时长: 43.4072 ms

## **代码说明**

| 路径             | 行数 | 内容                                |
| ---------------- | ---- | ----------------------------------- |
| src/model/stack_model.py| 44 | `output = (dcn_output1 + dcn_output2 + dcn_output3 + pnn_output + dcn_output1_ + dcn_output2_ + dcn_output3_ + dcn_output4_ + pnn_output_) / 9`|
| src/train/trainer.py | 567   | `batch_preds = self.model(interaction)` |

将融合的模型组成一个大模型,直接一起预测得出结果