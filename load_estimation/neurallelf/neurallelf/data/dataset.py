'''
A module for data loading and preparation.
'''
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Abstract dataset base class.
    """
    def __init__(self):
        self.pq_df = None
        self.v_df = None
        self.pflow_df = None
        self.qflow_df = None
        self.flow_present = False           #if flow data is loaded

        self.graph_df = None
        self.known_buses = None

    @abstractmethod
    def create_dataset(*args, **kwargs):
        pass