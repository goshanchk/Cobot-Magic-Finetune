from BeingH.dataset.datasets.vla_dataset import LeRobotIterableDataset


DATASET_REGISTRY = {
    "so101_posttrain": LeRobotIterableDataset,
}


DATASET_INFO = {
    "so101_posttrain": {
        "so101.pick_cube_plate": {
            "dataset_path": "/path/to/datasets/Being-H-EDU_SO101/pick_cube_plate_trimmed",
            "embodiment": "SO101",
            "embodiment_tag": "so101",
            "subtask": "so101.pick_cube_plate",
        },
    },
}
