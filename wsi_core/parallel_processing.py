# python peripherals
import multiprocessing.managers
import os
from multiprocessing import Process, Queue, Manager
from multiprocessing.managers import Namespace
import queue
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union, Generic, TypeVar, Optional, Dict, cast
import traceback
from enum import Enum, auto

# numpy
import numpy

# gipmed
from wsi_core.base import OutputObject


# =================================================
# ParallelProcessorBase Class
# =================================================
class ParallelProcessorBase(ABC):
    def __init__(self, num_workers: int, **kw: object):
        self._num_workers = num_workers
        self._workers = []
        self._is_processing = False
        self._manager = Manager()
        self._namespace = self._manager.Namespace()
        self._exempt_from_pickle = self._generate_exempt_from_pickle()
        super(ParallelProcessorBase, self).__init__(**kw)

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    def __getstate__(self):
        d = dict(self.__dict__)
        for key in self._exempt_from_pickle:
            del d[key]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def start(self):
        num_workers_digits = len(str(self._num_workers))

        print('Running Pre-Start')
        self._pre_start()

        self._add_shared_objects(namespace=self._namespace)
        args = [self._namespace] + self._get_args()

        self._workers = [Process(target=self._worker_func, args=tuple([worker_id] + args)) for worker_id in range(self._num_workers)]

        print('')
        for i, worker in enumerate(self._workers):
            worker.start()
            print(f'\rWorker Started {i+1:{" "}{"<"}{num_workers_digits}} / {self._num_workers:{" "}{">"}{num_workers_digits}}', end='')
        print('')

        self._is_processing = True

        print('Running Post-Start')
        self._post_start()

    def join(self):
        print('Running Pre-Join')
        self._pre_join()

        print('Joining processes')
        for worker in self._workers:
            worker.join()

        self._is_processing = False

        print('Running Post-Join')
        self._post_join()

    def _generate_exempt_from_pickle(self) -> List[str]:
        return ['_workers', '_manager', '_namespace']

    def _add_shared_objects(self, namespace: Namespace):
        pass

    def _get_args(self) -> List:
        return []

    @abstractmethod
    def _pre_start(self):
        pass

    @abstractmethod
    def _post_start(self):
        pass

    @abstractmethod
    def _pre_join(self):
        pass

    @abstractmethod
    def _post_join(self):
        pass

    @abstractmethod
    def _worker_func(self, **kwargs):
        pass


# =================================================
# ParallelProcessorTask Class
# =================================================
class ParallelProcessorTask(ABC):
    @abstractmethod
    def pre_process(self):
        pass

    @abstractmethod
    def process(self, namespace: Namespace):
        pass

    @abstractmethod
    def post_process(self):
        pass


# =================================================
# TaskParallelProcessor Class
# =================================================
class TaskParallelProcessor(ParallelProcessorBase, OutputObject):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, **kw)
        self._tasks_queue = Queue()
        self._completed_tasks_queue = Queue()
        self._tasks = self._generate_tasks()
        self._completed_tasks = []

    @property
    def tasks_count(self) -> int:
        return len(self._tasks)

    def _get_args(self) -> List:
        return [self._tasks_queue, self._completed_tasks_queue]

    def _generate_exempt_from_pickle(self) -> List[str]:
        exempt_from_pickle = super()._generate_exempt_from_pickle()
        exempt_from_pickle.append('_tasks')
        exempt_from_pickle.append('_completed_tasks')
        exempt_from_pickle.append('_tasks_queue')
        exempt_from_pickle.append('_completed_tasks_queue')
        return exempt_from_pickle

    def _pre_start(self):
        for task in self._tasks:
            self._tasks_queue.put(obj=task)

        for _ in range(self._num_workers):
            self._tasks_queue.put(obj=None)

    def _post_start(self):
        total_tasks_count = self.tasks_count + self._num_workers
        last_remaining_tasks_count = numpy.inf
        total_tasks_count_digits = len(str(total_tasks_count))
        while True:
            remaining_tasks_count = self._tasks_queue.qsize()
            if last_remaining_tasks_count > remaining_tasks_count:

                print(f'\rRemaining Tasks {remaining_tasks_count:{" "}{"<"}{total_tasks_count_digits}} / {total_tasks_count:{" "}{">"}{total_tasks_count_digits}}', end='')
                last_remaining_tasks_count = remaining_tasks_count

            if remaining_tasks_count == 0:
                break

        print('')

        print('Draining Queue')
        sentinels_count = 0
        while True:
            completed_task = self._completed_tasks_queue.get()
            if completed_task is None:
                sentinels_count = sentinels_count + 1
            else:
                self._completed_tasks.append(completed_task)

            if sentinels_count == self._num_workers:
                break

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_tasks(self) -> List[ParallelProcessorTask]:
        pass

    def _worker_func(self, worker_id: int, namespace: Namespace, tasks_queue: Queue, completed_tasks_queue: Queue):
        while True:
            task = cast(typ=ParallelProcessorTask, val=tasks_queue.get())
            if task is None:
                completed_tasks_queue.put(None)
                return

            try:
                task.pre_process()
                task.process(namespace=namespace)
                task.post_process()
            except:
                print()
                traceback.print_exc()

            completed_tasks_queue.put(task)


# =================================================
# BioMarker Class
# =================================================
class GetItemPolicy(Enum):
    Replace = auto()
    TryReplace = auto()


# =================================================
# OnlineParallelProcessor Class
# =================================================
class OnlineParallelProcessor(ParallelProcessorBase, OutputObject):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, **kw)
        self._items_queue_maxsize = items_queue_maxsize
        self._items_buffer_size = items_buffer_size
        self._items_buffer = []
        self._items_queue = self._manager.Queue(maxsize=items_queue_maxsize)
        self._sentinels_queue = self._manager.Queue()

    @abstractmethod
    def __getitem__(self, index: int) -> object:
        pass

    @abstractmethod
    def _generate_item(self, item_id: Optional[int], namespace: Namespace) -> object:
        pass

    def _add_stopper_sentinels(self):
        for worker_id in range(self._num_workers):
            self._sentinels_queue.put(item=None)

    def _worker_func(self, namespace: Namespace):
        while True:
            sentinel = None
            try:
                sentinel = self._sentinels_queue.get_nowait()
                if sentinel is None:
                    break
            except:
                pass

            try:
                item = self._generate_item(item_id=sentinel, namespace=namespace)
                self._items_queue.put(item=item)
            except:
                print()
                traceback.print_exc()


# =================================================
# InfiniteOnlineParallelProcessor Class
# =================================================
class InfiniteOnlineParallelProcessor(OnlineParallelProcessor):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_buffer_size: int, get_item_policy: GetItemPolicy, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, items_queue_maxsize=items_queue_maxsize, items_buffer_size=items_buffer_size, **kw)
        self._get_item_policy = get_item_policy

    def __getitem__(self, index: int) -> object:
        mod_index = numpy.mod(index, len(self._items_buffer))
        item = self._items_buffer[mod_index]

        new_item = None
        if self._get_item_policy == GetItemPolicy.TryReplace:
            try:
                new_item = self._items_queue.get_nowait()
            except queue.Empty:
                pass
        elif self._get_item_policy == GetItemPolicy.Replace:
            new_item = self._items_queue.get()

        if new_item is not None:
            rand_index = int(numpy.random.randint(self._items_buffer_size, size=1))
            self._items_buffer[rand_index] = new_item

        return item

    def stop(self):
        self._add_stopper_sentinels()

    def _pre_start(self):
        pass

    def _post_start(self):
        while len(self._items_buffer) < self._items_buffer_size:
            self._items_buffer.append(self._items_queue.get())
            print(f'\rBuffer Populated with {len(self._items_buffer)} Items', end='')
        print('')

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_item(self, item_id: Optional[int], namespace: Namespace) -> object:
        pass


# =================================================
# FiniteOnlineParallelProcessor Class
# =================================================
class FiniteOnlineParallelProcessor(OnlineParallelProcessor):
    def __init__(self, name: str, output_dir_path: Path, num_workers: int, items_queue_maxsize: int, items_count: int, **kw: object):
        super().__init__(name=name, output_dir_path=output_dir_path, num_workers=num_workers, items_queue_maxsize=items_queue_maxsize, **kw)
        self._items_count = items_count

        for item_id in range(items_count):
            self._sentinels_queue.put(item_id)

        self._add_stopper_sentinels()

    def __getitem__(self, index: int) -> object:
        return self._items_queue.get()

    def _pre_start(self):
        pass

    def _post_start(self):
        pass

    def _pre_join(self):
        pass

    def _post_join(self):
        pass

    @abstractmethod
    def _generate_item(self, item_id: Optional[int]) -> object:
        pass
