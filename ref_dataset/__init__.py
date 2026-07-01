from .refhlseg import RefHLSegDataset, collate_fn_medical


MEDICAL_DATASETS = {"refhlseg", "ref-lits", "ref-lidc", "qata-cov19", "mosmeddata+"}


def build_dataset(is_train, args):
    data_path = args.data_path
    split = "train" if is_train else args.test_split

    if args.data_set in MEDICAL_DATASETS:
        dataset = RefHLSegDataset(
            data_root=data_path,
            split=split,
            image_size=args.image_size,
            json_prefix=args.json_prefix,
        )
        dataset.collate_fn = collate_fn_medical
        return dataset

    raise ValueError(f"Invalid dataset: {args.data_set}")
