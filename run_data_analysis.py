from data_handling.data_loader import DataloaderKinderlabor

if __name__ == "__main__":
    loader_all = DataloaderKinderlabor()
    loader_all.plot_class_distributions()

    loader_orientation = DataloaderKinderlabor(task_type="ORIENTATION")
    loader_orientation.plot_class_distributions()
