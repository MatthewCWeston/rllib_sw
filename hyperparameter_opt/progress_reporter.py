"""
Borrowed and adapted from Neural Nova to give us a nicer format for our hyperparameter search.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Any, Literal
from datetime import datetime, timedelta

import pandas as pd
from ray.tune import ProgressReporter

from ray.tune.experiment.trial import _Location
from ray.tune.result import NODE_IP, PID
from ray.tune.utils import unflattened_lookup


if TYPE_CHECKING:
    from ray.tune.experiment import Trial


class CustomReporter(ProgressReporter):
    """
    Custom Progress Reporter for Ray Tune
    """

    def __init__(
        self,
        metric_columns: Optional[dict[str, str]] = None,
        parameter_columns: Optional[dict[str, str]] = None,
        max_report_frequency: int = 20,
        metric: Optional[str] = None,
        mode: Optional[Literal['min', 'max']] = 'max',
        include_location: bool = False,
        rounding: Optional[dict[str, int]] = None,
        time_col: Optional[str] = None
    ):
        """
        Initialize the custom reporter.

        :param metric_columns:
            Metric columns to include in the report.
        :param parameter_columns:
            Parameter columns to include in the report.
        :param max_report_frequency:
            Minimum time between console updates.
        :param metric:
            Single metric to sort trials by
        :param mode:
            Should we sort by min or max?
        :param rounding:
            Round specific columns to a specified precision
        :param time_col:
            Name of a column to be formatted as datetime.
        """
        self.__max_report_frequency = max_report_frequency
        self.__last_report_time = 0
        self.__last_update_time = 0
        self.__metric_columns = metric_columns or {}
        self.__parameter_columns = parameter_columns or {}
        self.__metric = metric
        self.__mode = mode
        self.__rounding = rounding
        self.__time_col = time_col

    def __get_trial_info(
        self,
        trial: Trial,
    ) -> dict[str, Any]:
        """
        Get the data for a single row of the stats table
        """
        result = trial.last_result
        config = trial.config
        info = {
            "Name": trial.trial_id,
            "Status": trial.status,
        }
        for k, v in self.__parameter_columns.items():
            info[v] = unflattened_lookup(k, config, default=None)
        for k, v in self.__metric_columns.items():
            info[v] = unflattened_lookup(k, result, default=None)
        return info

    def __create_stats_table(self, trials: list[Trial]) -> pd.DataFrame:
        """
        Create the main stats table, which is displayed in the console.
        This is limited at 20 rows to prevent crashing the program due to the
        curses bug described at the top of this file.
        """
        df = pd.DataFrame(
            data=[
                self.__get_trial_info(t) for t in trials
            ]
        )
        by = ['Status']
        ascending = [True]
        if self.__metric is not None:
            by.append(self.__metric)
            ascending.append(self.__mode == 'min')
        df.sort_values(by=by, ascending=ascending, inplace=True)
        df.reset_index(inplace=True, drop=True)
        if self.__rounding is not None:
            for key, value in self.__rounding.items():
                df[key] = df[key].round(value)
        if self.__time_col is not None:
            df[self.__time_col] = pd.to_timedelta(df[self.__time_col], unit='s')
        return df.iloc[0:20]

    def should_report(self, trials: list[Trial], done: bool = False) -> bool:
        """
        Called by Ray Tune to determine if we should update the console.
        If we return True, ray will eventually call our "report" method.
        """
        now = time.time()
        if now - self.__last_update_time > 2:
            self.__last_update_time = now
        if now - self.__last_report_time > self.__max_report_frequency:
            self.__last_report_time = now
            return True
        return done

    # noinspection PyUnresolvedReferences
    def report(self, trials: list[Trial], done: bool, *sys_info: dict):
        """
        Main function called by Ray Tune to refresh the window.
        """
        stats = self.__create_stats_table(trials)
        info = sys_info[0] + ' ' + sys_info[1]
        print(stats.__repr__())

