class Memory():

    def __init__(self,
                 max_tasks:int=500):
        self.max_tasks = max_tasks
        self.queue = []

    def enqueue(self, tasks):
        if isinstance(tasks, list):
            for task in tasks:
                if len(self.queue) == self.max_tasks:
                    return 0
                self.queue.append(task)
        else:
            if len(self.queue) == self.max_tasks:
                return 0
            self.queue.append(tasks)
        return 1

    def dequeue(self):
        if len(self.queue) > 0:
            task = self.queue.pop(0)
        else:
            task = None
        return task

    def empty(self):
        """Are there any tasks in the queue?"""
        return not len(self.queue)


    def usage(self):
        """Return number of tasks in memory"""
        return len(self.queue)


    def usage_perc(self):
        """Return the memory usage in percentage"""
        return self.usage()/self.max_tasks


    def within_budget(self, workload:int):
        """Compute if the workload is within memory budget"""
        return self.usage() + workload <= self.max_tasks


    def set_usage_perc(self, usage:float):
        """For testing purposes"""
        target_usage = round(usage * self.max_tasks)
        diff = target_usage - self.usage()
        if diff > 0:
            self.enqueue([None] * diff)
        elif diff < 0:
            self.queue = self.queue[abs(diff):]


    def reset(self):
        self.queue = []
