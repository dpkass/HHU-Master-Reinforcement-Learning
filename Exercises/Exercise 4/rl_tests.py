# author: Jan Robine
# date:   2023-04-25
# course: reinforcement learning

import collections
import numpy as np


def run_tests(generator):
    failed_msgs = collections.defaultdict(lambda: [])
    test_index = 0
    test_closed = True
    first = True

    def flush():
        nonlocal test_index, test_closed, first
        if not test_closed:
            test_index += 1
        num_tests = test_index
        if num_tests == 0:
            return
        num_failed = len(failed_msgs)
        num_passed = num_tests - num_failed
        print(f'{num_passed}/{num_tests} tests passed!')
        if num_failed > 0:
            for test_index, msgs in failed_msgs.items():
                if len(msgs) == 1:
                    print(f'Test #{test_index + 1} failed: {msgs[0]}')
                else:
                    print(f'Test #{test_index + 1} failed:')
                    for msg in msgs:
                        print(msg)
        
        failed_msgs.clear()
        test_index = 0
        test_closed = True
        first = False

    try:
        cmd = generator.send(None)
        while True:
            if isinstance(cmd, str):
                flush()
                if not first:
                    print()
                print(f'Testing {cmd}...')
                first = False
                cmd = generator.send(None)
            elif cmd is None:
                test_index += 1
                test_closed = True
                cmd = generator.send(None)
            else:
                test_closed = False
                assertion, msg = cmd
                if not assertion:
                    failed_msgs[test_index].append(msg)
                cmd = generator.send(assertion)
    except StopIteration:
        flush()


def check_numpy_array(x, name, shape=None, dtype=None):
    if not (yield isinstance(x, np.ndarray), f'{name} must be a NumPy array (got {type(x)})'):
        return False
    if shape is not None:
        if not (yield x.shape == shape, f'{name} must be a NumPy array with shape {tuple(shape)} (got {x.shape})'):
            return False
    if dtype is not None:
        if not (yield np.issubdtype(x.dtype, dtype), f'{name} must be a NumPy array of type {dtype.__name__} (got {x.dtype})'):
            return False
    return True
