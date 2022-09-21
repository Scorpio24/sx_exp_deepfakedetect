# deepfakes检测
## 目录结构
1. ./data: 保存数据的目录。
    * ./dataset：保存数据集的目录。
        * ./aug_frames：保存deepfakes_dataset.py中经过处理后的图片，用于验证处理效果。
        * ./manipulated_sequences：在utils.py中的get_method方法中用到，用目录的方式来保存使用到的修改方法。暂时没有用处，但为了程序正确运行而创建。
        * ./original_sequences：同上，保存原视频的方法。
        * ./test_set：测试集数据，具体结构见参考项目的readme。
        * ./training_set：训练集数据。具体结构见参考项目的readme。
        * ./validation_set：验证集数据。具体结构见参考项目的readme。
        * ./dfdc_test_labels.csv：保存测试集数据label的csv文件。
    * ./metadata：保存了所用数据的元数据。在本实验中为DFDC数据集的元数据。
        * ./metadata.json：DFDC数据的元数据文件。
2. ./models：保存模型相关的数据。
    * ./final_models：保存训练得到的最终模型的数据。
    * ./tests：保存测试模型的结果。
    * ./S3D_checkpoint_*：训练过程中在checkpoint保存的中间模型数据。
3. ./preprocessing：预处理程序目录，详见参考项目。
4. ./S3D：实验模型相关程序的目录。
    * ./configs：保存配置文件的目录。
    * ./transforms：保存transform工具程序的目录。
    * ./deepfakes_dataset.py：dataset类的程序文件。
    * ./get_masked_face_simple.py：简易方式得到掩码处理图片的程序文件。
    * ./get_masked_face.py：复杂但正式方式得到掩码处理图片的程序文件。
    * ./model.py：S3D模型的程序文件。
    * ./S3D-test.py：测试模型的程序文件。
    * ./S3D-train.py：训练模型的程序文件。
    * ./utils.py：工具方法的程序文件。
## 运行方法
在./SX_shiyan目录下运行，以保证程序正确处理目录。需要通过--config参数指定配置文件拿名称。
如：

    python3 ./S3D/S3D-train.py --config plan1
## 注意事项
实验中使用的albumentations库为新版本，而根据参考项目的environment.yml文件创建的环境中albumentations库为老版本，所以需要用如下命令升级：
    
    pip3 install --upgrade albumentations
        

