import os
import pickle


class TaskNotFoundError(Exception):
    pass


class Benchmark:

    def __init__(self, name, tasks, evaluator):
        self.name = name
        self.tasks = tasks
        self.evaluator = evaluator

    def get_tasks(self):
        return [task.name for task in self.tasks]

    def load_task(self, task_name):
        for task in self.tasks:
            if task.name == task_name:
                return task
        raise TaskNotFoundError(f"Task with name '{task_name}' not found.")

    def get_evaluator(self):
        return self.evaluator

    def save(self, folder_path, name):
        file_path = os.path.join(folder_path, f"{name}.pkl")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
