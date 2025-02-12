import time
import queue
import threading


class ThreadManager:
    def __init__(self, timeout=0.5):
        self.threads = []
        self.result = []
        self.queue = queue.Queue()
        self._num_items = 0
        self.timeout = timeout
        self._stop = False
        self._run_func = None

    def add_items(self, items):
        for item in items:
            self.add(item)

    def add(self, item):
        self.queue.put(item)
        self._num_items += 1

    @property
    def num_items(self):
        return self._num_items

    def update_result(self, item):
        self.result.append(item)
        self._num_items -= 1

    def _func_wrapper(self, thread_id, preserve_order):
        while not self.queue.empty():
            if self._stop:
                return
            item = self.queue.get(timeout=self.timeout)
            if preserve_order:
                self.result.append((thread_id, self._run_func(item)))
            else:
                self.result.append(self._run_func(item))
            self._num_items -= 1

    def run(self, func, num_threads=1, wait=True, preserve_order=False):
        assert isinstance(num_threads, int) and (num_threads == -1 or num_threads > 0)
        self._run_func = func
        num_threads_to_spawn = min(
            self.num_items, self.num_items if num_threads == -1 else num_threads
        )
        for i in range(num_threads_to_spawn):
            thread = threading.Thread(
                target=self._func_wrapper, args=(i, preserve_order)
            )
            self.threads.append(thread)
            thread.start()

        if wait:
            # Wait for all threads to complete
            for thread in self.threads:
                thread.join()
            return list(self.result) if not preserve_order else [
                thread_result
                for _, thread_result in sorted(self.result, key=lambda result: result[0])
            ]
        return self

    def stop(self):
        self._stop = True

    def speed(self, dur=1):
        start_num = self._num_items
        start_time = time.time()
        time.sleep(dur)
        return (start_num - self._num_items) / (time.time() - start_time)

    def is_alive(self):
        return any(thread.is_alive() for thread in self.threads)

    def __call__(self, func, items, num_threads=1, wait=True, preserve_order=False):
        self.add_items(items)
        return self.run(
            func, num_threads=num_threads, wait=wait,
            preserve_order=preserve_order
        )


def multi_thread(func, items, num_threads=4, preserve_order=True):
    if 0 <= num_threads <= 1:
        return [func(item) for item in items]
    return ThreadManager()(
        func, items, num_threads=int(num_threads),
        preserve_order=preserve_order, wait=True
    )
