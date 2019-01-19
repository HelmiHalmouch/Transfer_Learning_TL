# Transfer_Learning_TL using VGG16 and Resnet50
Developpement of transfert learning methods for custom image recognition <br/>
###Run the code 
run using the following command : python35 transfert_learning_vgg16.py <br/>
### Remarks
-To train the model we have used the '.fit' keras method and we have fixed the 'epochs=10' only for the test. <br/>
So you can try with other value higher than 10 to improve the accuracy of the model <br/>
-The saved trained model are in the forlder 'output_trained_model'
### datasets 
The dataset are in this path : Transfert_Learning_using_Resnet50/data.zip <br/>
(when you use tranfert learning based on vgg16, please move the folder data.zip in the same that of vgg16)
### Results
In the case oF TL using vgg16 architecture (with fine tuning):<br>
		-Training time is arround 73 minutes <br/>
		-loss=0.3300, accuracy: 97.5309% <br/>

In the case oF TL using resnet50 architecture (with fine tuning):<br>
		-Training time is arround 37 minutes <br/>
		-loss=0.1147, accuracy: 96.9136% <br/>

-The analysis of the training, loss, and validation as function of epoch numberare in: <br/>
Fig_train_loss_vs_val_loss.png and Fig_train_acc_vs_val_acc.png (see folder analysis_results) <br/>
- We should improve the computing time in the training step by using GPU
