"""
util.py
-------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""

import heapq
import inspect
import random
import signal
import sys
import time


class FixedRandom:
    def __init__(self):
        self.random = random.Random()
        self.random.seed('Pacman is pretty neat.')
        self.original_state = self.random.getstate()


# Data structures useful for implementing SearchAgents


class Stack:
    """A container with a last-in-first-out (LIFO) queuing policy."""

    def __init__(self):
        self.list = []

    def push(self, item):
        """Push 'item' onto the stack"""
        self.list.append(item)

    def pop(self):
        """Pop the most recently pushed item from the stack"""
        return self.list.pop()

    def is_empty(self):
        """Returns true if the stack is empty"""
        return len(self.list) == 0


class Queue:
    """A container with a first-in-first-out (FIFO) queuing policy."""

    def __init__(self):
        self.list = []

    def push(self, item):
        """Enqueue the 'item' into the queue"""
        self.list.insert(0, item)

    def pop(self):
        """
        De-queue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def is_empty(self):
        """Returns true if the queue is empty"""
        return len(self.list) == 0


class PriorityQueue:
    """
        Implements a priority queue data structure. Each inserted item
        has a priority associated with it and the client is usually interested
        in quick retrieval of the lowest-priority item in the queue. This
        data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def is_empty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild
        # the heap. If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class PriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """

    def __init__(self, priority_function):
        """priorityFunction (item) -> priority"""
        self.priority_function = priority_function  # store the priority function
        PriorityQueue.__init__(self)  # super-class initializer

    def push(self, item):
        """Adds an item to the queue with priority from the priority function"""
        PriorityQueue.push(self, item, self.priority_function(item))


def manhattan_distance(xy1, xy2):
    """Returns the Manhattan distance between points xy1 and xy2"""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


"""
  Data structures and functions useful for various course projects

  The search project should not need anything below this line.
"""


class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting data.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    return_val the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def increment_all(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.increment_all(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def arg_max(self):
        """
        Returns the key with the highest value.
        """
        if len(list(self.keys())) == 0:
            return None
        all_items = list(self.items())
        values = [x[1] for x in all_items]
        max_index = values.index(max(values))
        return all_items[max_index][0]

    def sorted_keys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sorted_keys()
        ['second', 'third', 'first']
        """
        sorted_items = list(self.items())
        sorted_items.sort(key=lambda x, y: sign(y[1] - x[1]))
        return [x[0] for x in sorted_items]

    def total_count(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.total_count())
        if total == 0:
            return

        for key in list(self.keys()):
            self[key] = self[key] / total

    def divide_all(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        total = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            total += x[key] * y[key]
        return total

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in list(y.items()):
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend


def raise_not_defined():
    file_name = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    raise Exception("*** Method not implemented: %s at line %s of %s" % (method, line, file_name))


def normalize(vector_or_counter):
    """
    normalize a vector or counter by dividing each value by the sum of all values
    """
    normalized_counter = Counter()
    if isinstance(vector_or_counter, type(Counter)):
        counter = vector_or_counter
        total = float(counter.total_count())

        if total == 0:
            return counter

        for key in list(counter.keys()):
            value = counter[key]
            normalized_counter[key] = value / total
        return normalized_counter

    vector = vector_or_counter
    s = float(sum(vector))
    if s == 0:
        return vector

    return [el / s for el in vector]


def n_sample(distribution, values, n):
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    rand = [random.random() for _ in range(n)]
    rand.sort()
    samples = []
    sample_pos, dist_pos, cdf = 0, 0, distribution[0]
    while sample_pos < n:
        if rand[sample_pos] < cdf:
            sample_pos += 1
            samples.append(values[dist_pos])
        else:
            dist_pos += 1
            cdf += distribution[dist_pos]
    return samples


def sample(distribution, values=None):
    if isinstance(distribution, Counter):
        items = sorted(distribution.items())
        distribution = [i[1] for i in items]
        values = [i[0] for i in items]
    if sum(distribution) != 1:
        distribution = normalize(distribution)
    choice = random.random()
    i, total = 0, distribution[0]
    while choice > total:
        i += 1
        total += distribution[i]
    return values[i]


def sample_from_counter(ctr):
    items = sorted(ctr.items())
    return sample([v for k, v in items], [k for k, v in items])


def get_probability(value, distribution, values):
    """
      Gives the probability of a value under a discrete distribution
      defined by (distributions, values).
    """
    total = 0.0
    for prob, val in zip(distribution, values):
        if val == value:
            total += prob
    return total


def flip_coin(p):
    r = random.random()
    return r < p


def choose_from_distribution(distribution):
    """Takes either a counter or a list of (prob, key) pairs and samples"""
    if isinstance(distribution, (dict, Counter)):
        return sample(distribution)
    r = random.random()
    base = 0.0
    for prob, element in distribution:
        base += prob
        if r <= base:
            return element


def nearest_point(pos):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    (current_row, current_col) = pos

    grid_row = int(current_row + 0.5)
    grid_col = int(current_col + 0.5)
    return grid_row, grid_col


def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if x >= 0:
        return 1

    return -1


def array_invert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for _ in array]
    for outer in array:
        for inner in enumerate(outer):
            i = inner[0]
            result[i].append(outer[i])
    return result


def matrix_as_list(matrix, value=True):
    """
    Turns a matrix into a list of coordinates matching the specified value
    """
    rows, cols = len(matrix), len(matrix[0])
    cells = []
    for row in range(rows):
        for col in range(cols):
            if matrix[row][col] == value:
                cells.append((row, col))
    return cells


def lookup(name, namespace):
    """
    Get a method or class from any imported module from its name.
    Usage: lookup(functionName, globals())
    """
    dots = name.count('.')
    if dots > 0:
        module_name, obj_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
        module = __import__(module_name)
        return getattr(module, obj_name)

    modules = [obj for obj in list(namespace.values()) if str(type(obj)) == "<type 'module'>"]
    options = [getattr(module, name) for module in modules if name in dir(module)]
    options += [obj[1] for obj in list(namespace.items()) if obj[0] == name]

    if len(options) == 1:
        return options[0]

    if len(options) > 1:
        raise Exception(f'Name conflict for {name}')

    raise Exception(f'{name} not found as a method or class')


def pause():
    """
    Pauses the output stream awaiting user feedback.
    """
    print("<Press enter/return to continue>")
    eval(input())


# code to handle timeouts
#
# FIXME
# NOTE: TimeoutFuncton is NOT reentrant.  Later timeouts will silently
# disable earlier timeouts.  Could be solved by maintaining a global list
# of active time outs.  Currently, questions which have test cases calling
# this have all student code so wrapped.
#


class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args, **keyArgs):
        # If we have SIGALRM signal, use it to cause an exception if and
        # when this function runs too long.  Otherwise check the time taken
        # after the method has returned, and throw an exception then.
        if hasattr(signal, 'SIGALRM'):
            old = signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.timeout)
            try:
                result = self.function(*args, **keyArgs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
        else:
            start_time = time.time()
            result = self.function(*args, **keyArgs)
            time_elapsed = time.time() - start_time
            if time_elapsed >= self.timeout:
                self.handle_timeout(None, None)
        return result


_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_MUTED = False


class WritableNull:
    def write(self, string):
        pass


def mute_print():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if _MUTED:
        return
    _MUTED = True

    _ORIGINAL_STDOUT = sys.stdout
    # _ORIGINAL_STDERR = sys.stderr
    sys.stdout = WritableNull()
    # sys.stderr = WritableNull()


def unmute_print():
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _MUTED
    if not _MUTED:
        return
    _MUTED = False

    sys.stdout = _ORIGINAL_STDOUT
    # sys.stderr = _ORIGINAL_STDERR
