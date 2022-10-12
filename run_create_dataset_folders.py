from data_loading import DataloaderKinderlabor
from utils import TaskType, DataSplit

if __name__ == "__main__":
    for data_split in DataSplit:
        for task_type in TaskType:
            loader_orientation = DataloaderKinderlabor(task_type=task_type, data_split=data_split)
