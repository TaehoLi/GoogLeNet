from dataset_MH import *

def dataset(NORMAL_DIR, FAULT_DIR, ADD_NORMAL_DIR, ADD_FAULT_DIR, input_size, batch_size, num_workers):
    # 카메라 번호 (1~4까지 있음, 카메라별로 각각 다 따로 학습해야됩니다.) CAM1 / CAM2 / CAM3 / CAM4
    cam1 = 'CAM1'
    cam2 = 'CAM2'
    cam3 = 'CAM3'
    cam4 = 'CAM4'
    
    # 기존 데이터 로드
    normal_list1 = filename_list(NORMAL_DIR, cam=cam1)
    fault_list1 = filename_list(FAULT_DIR, cam=cam1)
    normal_list2 = filename_list(NORMAL_DIR, cam=cam2)
    fault_list2 = filename_list(FAULT_DIR, cam=cam2)
    normal_list3 = filename_list(NORMAL_DIR, cam=cam3)
    fault_list3 = filename_list(FAULT_DIR, cam=cam3)
    normal_list4 = filename_list(NORMAL_DIR, cam=cam4)
    fault_list4 = filename_list(FAULT_DIR, cam=cam4)

    mhdb1 = MyungHwa_ansan_db(normal_list1, fault_list1, dataType='train', input_size=input_size, augmentation=True)
    mhdb2 = MyungHwa_ansan_db(normal_list2, fault_list2, dataType='train', input_size=input_size, augmentation=True)
    mhdb3 = MyungHwa_ansan_db(normal_list3, fault_list3, dataType='train', input_size=input_size, augmentation=True)
    mhdb4 = MyungHwa_ansan_db(normal_list4, fault_list4, dataType='train', input_size=input_size, augmentation=True)

    mhdb5 = MyungHwa_ansan_db(normal_list1, fault_list1, dataType='val', input_size=input_size, augmentation=True)
    mhdb6 = MyungHwa_ansan_db(normal_list2, fault_list2, dataType='val', input_size=input_size, augmentation=True)
    mhdb7 = MyungHwa_ansan_db(normal_list3, fault_list3, dataType='val', input_size=input_size, augmentation=True)
    mhdb8 = MyungHwa_ansan_db(normal_list4, fault_list4, dataType='val', input_size=input_size, augmentation=True)

    mhdb9 = MyungHwa_ansan_db(normal_list1, fault_list1, dataType='test', input_size=input_size, augmentation=True)
    mhdb10 = MyungHwa_ansan_db(normal_list2, fault_list2, dataType='test', input_size=input_size, augmentation=True)
    mhdb11 = MyungHwa_ansan_db(normal_list3, fault_list3, dataType='test', input_size=input_size, augmentation=True)
    mhdb12 = MyungHwa_ansan_db(normal_list4, fault_list4, dataType='test', input_size=input_size, augmentation=True)

    # 추가 데이터 로드
    add_normal_list1 = filename_list(ADD_NORMAL_DIR, cam=cam1)
    add_fault_list1 = filename_list(ADD_FAULT_DIR, cam=cam1)
    add_normal_list2 = filename_list(ADD_NORMAL_DIR, cam=cam2)
    add_fault_list2 = filename_list(ADD_FAULT_DIR, cam=cam2)
    add_normal_list3 = filename_list(ADD_NORMAL_DIR, cam=cam3)
    add_fault_list3 = filename_list(ADD_FAULT_DIR, cam=cam3)
    add_normal_list4 = filename_list(ADD_NORMAL_DIR, cam=cam4)
    add_fault_list4 = filename_list(ADD_FAULT_DIR, cam=cam4)

    add_mhdb1 = MyungHwa_ansan_db(add_normal_list1, add_fault_list1, dataType='train', input_size=input_size, augmentation=True)
    add_mhdb2 = MyungHwa_ansan_db(add_normal_list2, add_fault_list2, dataType='train', input_size=input_size, augmentation=True)
    add_mhdb3 = MyungHwa_ansan_db(add_normal_list3, add_fault_list3, dataType='train', input_size=input_size, augmentation=True)
    add_mhdb4 = MyungHwa_ansan_db(add_normal_list4, add_fault_list4, dataType='train', input_size=input_size, augmentation=True)

    add_mhdb5 = MyungHwa_ansan_db(add_normal_list1, add_fault_list1, dataType='val', input_size=input_size, augmentation=True)
    add_mhdb6 = MyungHwa_ansan_db(add_normal_list2, add_fault_list2, dataType='val', input_size=input_size, augmentation=True)
    add_mhdb7 = MyungHwa_ansan_db(add_normal_list3, add_fault_list3, dataType='val', input_size=input_size, augmentation=True)
    add_mhdb8 = MyungHwa_ansan_db(add_normal_list4, add_fault_list4, dataType='val', input_size=input_size, augmentation=True)

    add_mhdb9 = MyungHwa_ansan_db(add_normal_list1, add_fault_list1, dataType='test', input_size=input_size, augmentation=True)
    add_mhdb10 = MyungHwa_ansan_db(add_normal_list2, add_fault_list2, dataType='test', input_size=input_size, augmentation=True)
    add_mhdb11 = MyungHwa_ansan_db(add_normal_list3, add_fault_list3, dataType='test', input_size=input_size, augmentation=True)
    add_mhdb12 = MyungHwa_ansan_db(add_normal_list4, add_fault_list4, dataType='test', input_size=input_size, augmentation=True)

    # 데이터셋 합치기
    concat_mhdb1 = torch.utils.data.ConcatDataset([mhdb1, add_mhdb1])
    concat_mhdb2 = torch.utils.data.ConcatDataset([mhdb2, add_mhdb2])
    concat_mhdb3 = torch.utils.data.ConcatDataset([mhdb3, add_mhdb3])
    concat_mhdb4 = torch.utils.data.ConcatDataset([mhdb4, add_mhdb4])

    concat_mhdb5 = torch.utils.data.ConcatDataset([mhdb5, add_mhdb5])
    concat_mhdb6 = torch.utils.data.ConcatDataset([mhdb6, add_mhdb6])
    concat_mhdb7 = torch.utils.data.ConcatDataset([mhdb7, add_mhdb7])
    concat_mhdb8 = torch.utils.data.ConcatDataset([mhdb8, add_mhdb8])

    concat_mhdb9 = torch.utils.data.ConcatDataset([mhdb9, add_mhdb9])
    concat_mhdb10 = torch.utils.data.ConcatDataset([mhdb10, add_mhdb10])
    concat_mhdb11 = torch.utils.data.ConcatDataset([mhdb11, add_mhdb11])
    concat_mhdb12 = torch.utils.data.ConcatDataset([mhdb12, add_mhdb12])


    #print("Train size:", concat_mhdb1.__getitem__(0)[0].size(), concat_mhdb1.__len__())
    #print("Val size:", concat_mhdb5.__getitem__(0)[0].size(), concat_mhdb5.__len__())
    #print("Test size:", concat_mhdb9.__getitem__(0)[0].size(), concat_mhdb9.__len__())

    mhdb_batch1 = torch.utils.data.DataLoader(concat_mhdb1, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mhdb_batch2 = torch.utils.data.DataLoader(concat_mhdb2, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mhdb_batch3 = torch.utils.data.DataLoader(concat_mhdb3, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mhdb_batch4 = torch.utils.data.DataLoader(concat_mhdb4, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_mhdb_batch1 = torch.utils.data.DataLoader(concat_mhdb5, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_mhdb_batch2 = torch.utils.data.DataLoader(concat_mhdb6, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_mhdb_batch3 = torch.utils.data.DataLoader(concat_mhdb7, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_mhdb_batch4 = torch.utils.data.DataLoader(concat_mhdb8, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_mhdb_batch1 = torch.utils.data.DataLoader(concat_mhdb9, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_mhdb_batch2 = torch.utils.data.DataLoader(concat_mhdb10, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_mhdb_batch3 = torch.utils.data.DataLoader(concat_mhdb11, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_mhdb_batch4 = torch.utils.data.DataLoader(concat_mhdb12, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return [mhdb_batch1, mhdb_batch2, mhdb_batch3, mhdb_batch4,
            val_mhdb_batch1, val_mhdb_batch2, val_mhdb_batch3, val_mhdb_batch4,
            test_mhdb_batch1, test_mhdb_batch2, test_mhdb_batch3, test_mhdb_batch4]