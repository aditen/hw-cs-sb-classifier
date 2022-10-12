from data_loading import DataloaderKinderlabor
from utils import TaskType, DataSplit, data_split_dict, long_names_tasks
from visualizing import VisualizerKinderlabor

if __name__ == "__main__":
    for data_split in DataSplit:
        for task_type in TaskType:
            loader = DataloaderKinderlabor(task_type=task_type, data_split=data_split)
            visualizer = VisualizerKinderlabor(loader,
                                               run_id=f"data_split[{data_split_dict[data_split]}]_task[{long_names_tasks[task_type]}]")
            visualizer.visualize_class_distributions()
            visualizer.visualize_some_train_samples()
