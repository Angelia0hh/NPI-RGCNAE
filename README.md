# NPI-RGCNAE
  In this work, we used Relational Graph Convolutional 		Network Auto-Encoder(RGCNAE) to predict the interactions between ncRNA and protein.
  Under the `'data'` folder, there are two folders `'raw_data'` and `'generated_data'`, which provide the raw dataset and the data generated for our method, respectively.
  `'src/dataset_settings.ini'` is a parameter configuration file for each dataset.
# How to run
The program is written in **Python 3.7** and to run the code we provide, you need to install the `requirements.txt` through inputting the following command in command line mode:

```bash
pip install -r requirements.txt 
```

And use the below command to run the `main.py`:

```bash
python src/main.py -method {method name} -dataset {dataset name} -negative_random_sample {negative generation name} -layers {the number of layers} -with_side_information {side information}
```
The meanings of parameters: 

|  Parameter | Optional value |Meaning|
|--|--|--|
| **method name** | single_dataset_prediction |<p align="left">Predict ncRNA-protein interactions on a dataset chosed by the dataset parameter. Other parameters are the same as your input.<br> For example,`python src/main.py -method single_dataset_prediction -dataset RPI369  -negative_random_sample sort -layers 1 -with_side_information False`</p>|
| |compare_different_combinations|<p align="left">Compare the performance of "with sequence-based features" and "without sequence-based method" on RPI369, RPI2241,RPI7317 and NPInter10412.</p>|
| |compare_different_layers |<p align="left">Compare the performance of R-GCN layers varying from 1 to 4 on RPI369, RPI2241,RPI7317 and NPInter10412.</p>
| |compare_negative_sample_methods |<p align="left">Compare the performance of three different negative sample generation methods on RPI369, RPI2241,RPI7317 and NPInter10412.</p>
| |timeAnalysis|<p align="left">Calculate the runnning time on different datasets. You only need to enter two parameters, "dataset" and "negative_random_sample". <br>The "dataset" can be benchmarking datasets, such as  RPI7317, RPI369, RPI2241, NPInter_4158 and NPInter_10412. It can also be datasets randomly selected from NPInter10412, such as NPInter_10412_0.2,   NPInter_10412_0.4, NPInter_10412_0.6, NPInter_10412_0.8. <br>For RPI7317, RPI369, RPI2241, NPInter_4158 and NPInter_10412 datasets, the input parameter 'negative_random_sample' can be 'sort', 'sort_random', 'random'. <br>For NPInter_10412_0.2, NPInter_10412_0.4, NPInter_10412_0.6, and NPInter_10412_0.8, the input parameter 'negative_random_sample' can be the randomly sampling proportion, 0.2,0.4,0.6, and 0.8.<br>For example,`python src/main.py -method timeAnalysis -dataset  NPInter_10412_0.4 -negative_random_sample 0.4`</p>
|**dataset name** |	<br>RPI369</br> 	<br>RPI2241</br> 	<br>RPI7317</br><br> NPInter_10412</br><br> NPInter_4158</br> |
|**negative_generation name** |<br>random</br><br>sort_random</br><br>sort</br>|<p align="left">Please refer to our paper for the specific meaning of the above parameters.</p> 
|**the number of layers** |1,2,3,4|<p align="left">The number of R-GCN layers.</p>
|**side information**|True or False|<p align="left">Means whether use side information as part of the node feature.</p>

For "compare_different_combinations", "compare_different_layers", and "	compare_negative_sample_methods", the parameters are default and no additional input is required.




