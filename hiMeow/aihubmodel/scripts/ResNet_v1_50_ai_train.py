from ResNet_ops_1 import *



train_path = 'C:/Users/jjj53/Desktop/himeow/dataset/Training/' #경로 마지막에 반드시 '/'를 기입해야합니다.
model_name = 'resnet_v1_50'
epoch = 50

if __name__ == '__main__':
    fine_tunning = Fine_tunning(train_path=train_path,
                                model_name=model_name,
                                epoch=epoch)
    history = fine_tunning.training()
    fine_tunning.save_accuracy(history)