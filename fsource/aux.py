"""Auxiliary routines"""
from __future__ import print_function
import time

class Stopwatch:
    def __init__(self, time=time.time):
        self.time = time
        self.previous = time()
        self.initial = self.previous

    def click(self):
        elapsed = self.time() - self.previous
        self.previous += elapsed
        return elapsed

    def total(self):
        return self.time() - self.initial

