from apps.ml_models.LGTDM.data_provider.data_loader import QARDataset
from torch.utils.data import DataLoader


# 数据集创建器
def data_provider(args, mode):

    # 数据集Dataset字典
    data_dict = {
        "QARDataset": QARDataset,
    }

    # 数据集Dataset
    Dataset = data_dict[args.dataset_type]

    # 根据模式设置 shuffle_flag 和 batch_size
    if mode == "train":
        shuffle_flag = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        batch_size = args.batch_size
    
    # 创建数据集
    data_set = Dataset(
        seq_len=args.seq_len,
        split_len=args.split_len,
        mode=mode,
        missing_ratio=args.missing_ratio,
        scale=args.scale,
        random_seed=args.random_seed,
    )

    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
    )

    return data_set, data_loader