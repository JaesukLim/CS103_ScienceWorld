import json
import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))
JAR_FILE = 'scienceworld.jar'
JAR_PATH = os.path.join(BASEPATH, JAR_FILE)
TASK_FILE = 'tasks.json'
TASK_PATH = os.path.join(BASEPATH, TASK_FILE)


def is_in_debug_mode() -> bool:
    """Determine whether debug mode should be enabled.

    Debug mode sends JAR output to the console and allows the user to attach a debugger at port 5005.

    To enable debug mode, set the environment variable SCIENCEWORLD_DEBUG to "1" or "true".
    """
    if "SCIENCEWORLD_DEBUG" not in os.environ:
        return False
    env_value = os.environ["SCIENCEWORLD_DEBUG"].lower()
    if env_value in {"1", "true"}:
        return True
    elif env_value in {"", "0", "false"}:
        return False
    else:
        raise ValueError(f'{env_value!r} is not a valid value for "SCIENCEWORLD_DEBUG"')


DEBUG_MODE = is_in_debug_mode()

with open(TASK_PATH) as file:
    TASKS = json.load(file)

VISIBLE_TASKS = [task for task in TASKS if task.get("visible_in_task_list", True)]

ID2TASK = {task['task_id']: task['task_name'] for task in TASKS}
ID2INTERNAL_TASK = {task['task_id']: task.get('internal_task_name', task['task_name']) for task in TASKS}
VISIBLE_ID2TASK = {task['task_id']: task['task_name'] for task in VISIBLE_TASKS}
TASKNAME2ID = {task['task_name']: task['task_id'] for task in TASKS}
TASKNAME2INTERNAL = {task['task_name']: task.get('internal_task_name', task['task_name']) for task in TASKS}
# Names used in the paper
NAME2ID = {f"{task['topic']} {task['task']}": task['task_id'] for task in TASKS}
