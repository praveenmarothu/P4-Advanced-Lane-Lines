import collections

class GlobalVars(object):

    global_left_points_x = collections.deque(maxlen=10)
    global_left_points_y = collections.deque(maxlen=10)
    global_right_points_x = collections.deque(maxlen=10)
    global_right_points_y = collections.deque(maxlen=10)
    global_left_poly = None
    global_right_poly = None
