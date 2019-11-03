
import numpy as np
from math import pow
from math import sqrt
import matplotlib.pyplot as plt

class trainData(object):
    def __init__(self, user, item, ratio):
        self.user = user
        self.item = item
        self.ratio = ratio
