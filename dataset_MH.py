import numpy as np
import os
import cv2

from PIL import Image

import torch
import torchvision

# cam CAM1, CAM2
def filename_list(filePath='../../../snapshots_pureok', cam=None):
    """
    학습에 사용될 이미지 데이터들의 경로를 list형태로 만들어주는 함수입니다.
    
    filePath : 데이터가 저장된 디렉토리
    cam : 카메라 번호 ('CAM1', 'CAM2', 'CAM3', 'CAM4' 중 하나가 들어감)
    """
    snapshot_dir = np.array(os.listdir(filePath))
    snapshot_dir = snapshot_dir[[os.path.isdir(os.path.join(filePath, i)) for i in snapshot_dir]]
    file_list = [os.path.join(filePath, i, j) for i in snapshot_dir for j in os.listdir(os.path.join(filePath, i))]
    if cam == None:
        file_list = [i for i in file_list if i[-5:] != 'masks']
    else:
        # 찍힌 카메라 별로 데이터를 뽑아낼 수 있게 하드코딩됨
        file_list = [i for i in file_list if i[-31:-27] == cam]
    
    return file_list

# train, val, test 나누는 방법을 비율로 변경
class MyungHwa_ansan_db(torch.utils.data.Dataset):
    def __init__(self, normal_data_list, fault_data_list, dataType='train', input_size=(512,512), augmentation=True, testSize_ok=0.3, valSize_ok=0.1, testSize_ng=0.3, valSize_ng=0.1):
        super(MyungHwa_ansan_db, self).__init__()
        """
        명화공업 데이터 경로들의 list를 입력으로 받는 pytorch 데이터셋 입니다.
        
        normal_data_list : OK 데이터에 대한 filename_list결과를 input으로 넣어주면 됩니다.
        fault_data_list : NG 데이터에 대한 filename_list결과를 input으로 넣어주면 됩니다.
        dataType : 데이터 셋의 종류를 결정하는 파라미터입니다. ('train', 'val', 'test' 중 하나가 들어감)
        input_size : 학습, 테스트에 사용될 이미지의 크기입니다. tuple형태로 입력하면 되고, height, width순서입니다.
        뒤쪽 파라미터는 건드리지 않으셔도 됩니다.
        """
        assert dataType in ['train', 'val', 'test']
        ok_length = len(normal_data_list)
        ng_length = len(fault_data_list)
        
        self.TEST_SIZE_OK = round(ok_length * testSize_ok) # for each label
        self.VAL_SIZE_OK = round(ok_length * valSize_ok) # for each label
        self.TEST_SIZE_NG = round(ng_length * testSize_ng) # for each label
        self.VAL_SIZE_NG = round(ng_length * valSize_ng) # for each label
    
        self.cats = ['NORMAL', 'FAULT']
        if dataType == 'train':
            self.filename_list = normal_data_list[:-self.TEST_SIZE_OK-self.VAL_SIZE_OK] + fault_data_list[:-self.TEST_SIZE_NG-self.VAL_SIZE_NG]
            self.label_list = [0 for i in normal_data_list[:-self.TEST_SIZE_OK-self.VAL_SIZE_OK]] + [1 for i in fault_data_list[:-self.TEST_SIZE_NG-self.VAL_SIZE_NG]]
        elif dataType == 'val':
            self.filename_list = normal_data_list[-self.TEST_SIZE_OK-self.VAL_SIZE_OK:-self.TEST_SIZE_OK] + fault_data_list[-self.TEST_SIZE_NG-self.VAL_SIZE_NG:-self.TEST_SIZE_NG]
            self.label_list = [0 for i in normal_data_list[-self.TEST_SIZE_OK-self.VAL_SIZE_OK:-self.TEST_SIZE_OK]] + [1 for i in fault_data_list[-self.TEST_SIZE_NG-self.VAL_SIZE_NG:-self.TEST_SIZE_NG]]
        elif dataType == 'test':
            self.filename_list = normal_data_list[-self.TEST_SIZE_OK:] + fault_data_list[-self.TEST_SIZE_NG:]
            self.label_list = [0 for i in normal_data_list[-self.TEST_SIZE_OK:]] + [1 for i in fault_data_list[-self.TEST_SIZE_NG:]]
        
        transform_list = []
        transform_list.append(torchvision.transforms.ToPILImage())
        transform_list.append(torchvision.transforms.Resize(size=input_size))
        if augmentation:
            randomcrop_size = (input_size[0] - 40, input_size[1] - 40)
            transform_list.append(torchvision.transforms.RandomApply([torchvision.transforms.RandomCrop(size=randomcrop_size)], p=0.5))
            transform_list.append(torchvision.transforms.Resize(size=input_size))
        transform_list.append(torchvision.transforms.ToTensor())
        
        self.transform = torchvision.transforms.Compose(transform_list)
        
    def __getitem__(self, index):
        imgPath = self.filename_list[index]
        # Image
        #img = Image.open(imgPath)
        #img = img.convert('RGB')
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        # Label
        label = self.label_list[index]
        label = torch.tensor(label, dtype=torch.float)
        
        return img, label
        
    def __len__(self):
        return len(self.filename_list)