def dataload_balanced(data_files):
    # 데이터셋을 생성하는 사용자 정의 Dataset 클래스
    class CustomDataset(Dataset):
        def __init__(self, data_files):
            self.data_files = data_files
            # 여기서 데이터 파일을 로드하는 로직을 추가하세요.
            # 예를 들어, 이미지 파일을 로드하거나 텍스트 파일을 읽는 등의 작업을 수행합니다.

        def __len__(self):
            return len(self.data_files)

        def __getitem__(self, idx):
            # 데이터 파일을 읽고 반환하는 로직을 추가하세요.
            # 예를 들어, self.data_files[idx]를 사용하여 파일을 읽습니다.
            return self.data_files[idx]  # 실제 데이터 반환 로직으로 수정 필요

    # 사용자 정의 데이터셋 생성
    input_dataset = CustomDataset(data_files)
    
    # 데이터셋 나누기
    train_size = int(0.8 * len(input_dataset))
    valid_size = len(input_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(input_dataset, [train_size, valid_size])
    
    print(f"학습 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(valid_dataset)}")

    return train_dataset, valid_dataset

# 사용 예시
data_files = ['file1.txt', 'file2.txt', 'file3.txt']  # 데이터 파일 목록
train_dataset, valid_dataset = dataload_balanced(data_files)
