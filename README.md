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
| **method name** | single_dataset_prediction |predict ncRNA-protein interactions on a dataset chosed by the dataset parameter.|
| |compare_different_combinations|compare the performance of *"node embeddings + k-mer"* and *"node embeddings"* two different feature combinations on RPI369, RPI2241,RPI7317 and NPInter10412.|
| |compare_different_layers |compare the performance of R-GCN layers varying from 1 to 4 on RPI369, RPI2241,RPI7317 and NPInter10412.
| |compare_negative_sample_methods |compare the performance of three different negative sample generation methods on RPI369, RPI2241,RPI7317 and NPInter10412.
|**dataset name** |	<br>RPI369</br> 	<br>RPI2241</br> 	<br>RPI7317</br><br> NPInter_10412</br><br> NPInter_4158</br> |
|**negative_generation name** |<br>random</br><br>sort_random</br><br>sort</br>|Please refer to our paper for the specific meaning of the above parameters. 
|**the number of layers** |1,2,3,4|the number of R-GCN layers
|**side information**|True or False|means whether use side information as part of the node feature.

*For "single_dataset_prediction", parameters are the same as your input, while for other three methods, the parameters are default and no additional input is required.

For example,

```bash
>python src/main.py -method single_dataset_prediction -dataset RPI369  -negative_random_sample sort_random -layers 1 -with_side_information True
