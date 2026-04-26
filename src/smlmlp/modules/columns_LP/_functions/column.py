#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author        : Lancelot PINCET
# GitHub        : https://github.com/LancelotPincet


# %% Libraries
from contextlib import nullcontext

import numpy as np

from smlmlp import columns, DataFrame, MainDataFrame, LocsReceiver


# %% Class
class column:
    """
    Decorator/descriptor used to declare dataframe columns.

    Parameters
    ----------
    headers : list[str]
        Possible file headers for this column. The first one is the canonical header.
    dtype : type
        Numpy dtype used when assigning this column.
    save : bool, default=True
        Whether this column should be saved.
    agg : str or callable, default='mean'
        Aggregation method used when merging from a parent dataframe.
    index : str or bool, default=False
        Parent dataframe name for index columns. False means this is not an index column.
    """

    def __init__(self, *, headers, dtype, save=True, agg="mean", index=False):
        """Initialize the object."""
        self.headers = headers
        self.header = headers[0]
        self.save = save
        self.agg = agg
        self.index = index
        self.dtype = dtype

    def __call__(self, func):
        """Decorator entry point."""
        self.func = func
        self.col = func.__name__

        if self.col in columns:
            raise SyntaxError(f"Column {self.col} is defined twice")
        columns[self.col] = self

        for header in self.headers:
            if header in columns.headers:
                raise SyntaxError(f"Header {header} is defined twice")
            columns.headers[header] = self

        return self

    def __set_name__(self, cls, name):
        """Called when the descriptor is assigned to a class."""
        self.cls = cls
        self.df_name = cls.__name__

        if self.index:
            if self.df_name != f"{self.header}s":
                raise SyntaxError(
                    f"Index column {self.col} does not coincide with DataFrame name {self.df_name}"
                )
            cls.index_header = self.header
            cls.index_col = self.col
            cls.parent_name = self.index
            cls.df_name = self.df_name

        if not hasattr(cls, "columns_dict"):
            cls.columns_dict = {}
        cls.columns_dict[self.col] = self

        if self.save:
            save_cls = MainDataFrame if self.index else cls
            if not hasattr(save_cls, "head2save"):
                save_cls.head2save = []
            for header in self.headers:
                if header not in save_cls.head2save:
                    save_cls.head2save.append(header)

        @property
        def _col(instance):
            """Return col."""
            return self

        setattr(cls, f"_{self.col}", _col)

        if self.index or issubclass(cls, MainDataFrame):
            setattr(MainDataFrame, f"{self.col}_mine", True)
            setattr(DataFrame, f"{self.col}_mine", False)
        else:
            setattr(MainDataFrame, f"{self.col}_mine", False)
            setattr(DataFrame, f"{self.col}_mine", True)

        self._install_dataframe_exists_property()
        self._install_dataframe_merge_property()

        if not issubclass(cls, MainDataFrame):
            if self.index:
                self._install_index_property_on_main()
                self._install_locsreceiver_dataframe_access()
            else:
                self._install_main_dataframe_spread_property()

    # Internal helpers

    def _printer_context(self, df, message):
        """Handle printer context."""
        printer = getattr(df.locs, "printer", None)
        return printer.timeit(message) if printer is not None else nullcontext()

    def _coerce_array(self, value):
        """Handle coerce array."""
        return np.asarray(value, dtype=self.dtype)

    def _drop_column(self, df, header):
        """Handle drop column."""
        df.drop(columns=[header], inplace=True, errors="ignore")

    def _ensure_column_materialized(
        self, df, col_name, header_name=None, allow_none=False
    ):
        """
        Ensure a logical column exists physically in df.columns.

        Parameters
        ----------
        df : dataframe-like
        col_name : str
            Descriptor/property name.
        header_name : str or None
            Physical header name to check. If None, uses col_name.
        allow_none : bool, default=False
            If True, return False instead of raising when the column cannot
            be materialized.

        Returns
        -------
        bool
            True if the physical header exists after this call, else False.
        """
        if header_name is None:
            header_name = col_name

        if header_name in df.columns:
            return True

        if header_name == getattr(df, "index_header", None):
            return True

        value = getattr(df, col_name, None)
        if value is None:
            if allow_none:
                return False
            raise ValueError(
                f"{col_name} not in {getattr(df, 'df_name', df.__class__.__name__)} and cannot be materialized"
            )

        if header_name not in df.columns:
            setattr(df, col_name, value)

        if header_name in df.columns:
            return True

        if allow_none:
            return False

        raise ValueError(
            f"{col_name} not in {getattr(df, 'df_name', df.__class__.__name__)} and cannot be materialized"
        )

    def _series_for_header(self, df, header):
        """Return a Series for a physical column or dataframe index header."""
        if header in df.columns:
            return df[header]
        if header == getattr(df, "index_header", None):
            return df.index.to_series(index=df.index)
        return None

    def _aggregate_parent_to_child(self, parent, child, value_header, allow_none=False):
        """
        Aggregate parent[value_header] into child rows by child.index_header,
        aligned explicitly on child.index.
        """
        value = self._series_for_header(parent, value_header)
        if value is None:
            if allow_none:
                return None
            raise ValueError(
                f"{value_header} not in {child.parent_name} and cannot be merged into {child.df_name}"
            )

        group_key = self._series_for_header(parent, child.index_header)
        if group_key is None:
            if allow_none:
                return None
            raise ValueError(
                f"{child.index_header} not in {child.parent_name} and cannot be used to merge {value_header} into {child.df_name}"
            )

        grouped = value.groupby(group_key).agg(self.agg)
        return grouped.reindex(child.index).to_numpy()

    def _map_child_to_detections(self, child, value_header, allow_none=False):
        """
        Spread child[value_header] into detections using detections[child.index_header].
        """
        dets = child.locs.detections

        if child.index_header not in dets.columns:
            materialized = self._ensure_column_materialized(
                dets,
                child.index_col,
                child.index_header,
                allow_none=allow_none,
            )
            if not materialized:
                return None

        if value_header not in child.columns:
            if allow_none:
                return None
            raise ValueError(
                f"{value_header} not in {child.df_name} and cannot be spread into detections"
            )

        mapped = dets[child.index_header].map(child[value_header])
        missing = mapped.isna() & dets[child.index_header].notna()
        missing &= dets[child.index_header] != 0
        if missing.any():
            missing_ids = sorted(set(dets.loc[missing, child.index_header].tolist()))
            if allow_none:
                return None
            raise ValueError(
                f"Missing {child.df_name} rows for {child.index_header} ids {missing_ids}"
            )

        return mapped.to_numpy()

    def _exists_dataframe(self, df):
        """Return the dataframe that physically owns this column for exists checks."""
        locs = getattr(df, "locs", None)
        df_dict = getattr(locs, "df_dict", None)

        if self.index:
            if df_dict is not None:
                return df_dict.get("detections")
            return getattr(locs, "detections", None)

        if getattr(df, "df_name", None) == self.df_name:
            return df

        if df_dict is None:
            return None
        return df_dict.get(self.df_name)

    # Installation methods

    def _install_dataframe_exists_property(self):
        """Install a passive physical-column existence property on all dataframes."""
        exists_name = f"{self.col}_exists"
        if hasattr(MainDataFrame, exists_name) or hasattr(DataFrame, exists_name):
            raise SyntaxError(f"{exists_name} cannot be defined twice")

        @property
        def col_exists(df):
            """Return whether the owning dataframe physically contains the header."""
            exists_df = self._exists_dataframe(df)
            return exists_df is not None and self.header in exists_df.columns

        setattr(MainDataFrame, exists_name, col_exists)
        setattr(DataFrame, exists_name, col_exists)

    def _install_dataframe_merge_property(self):
        """
        Install a merge property on DataFrame for this column name.

        This allows intermediate dataframes (e.g. points) to lazily aggregate
        a column from their parent dataframe when possible.
        """
        if hasattr(DataFrame, self.col):
            raise SyntaxError(f"{self.col} cannot be defined twice in DataFrame")

        @property
        def merged_col(df):
            """Return merged col."""
            with self._printer_context(
                df, f"merging {self.header} from {df.parent_name} into {df.df_name}"
            ):
                if self.header not in df.columns:
                    parent = getattr(df.locs, df.parent_name)
                    if parent is None:
                        return None

                    if self.header not in parent.columns:
                        if not self._ensure_column_materialized(
                            parent,
                            self.col,
                            self.header,
                            allow_none=True,
                        ):
                            return None

                    if df.index_header not in parent.columns:
                        if not self._ensure_column_materialized(
                            parent,
                            df.index_col,
                            df.index_header,
                            allow_none=True,
                        ):
                            return None

                    merged = self._aggregate_parent_to_child(
                        parent=parent,
                        child=df,
                        value_header=self.header,
                        allow_none=True,
                    )
                    if merged is None:
                        return None

                    df[self.header] = merged

                return df[self.header].to_numpy()

        @merged_col.setter
        def merged_col(df, value):
            """Set merged col."""
            with self._printer_context(
                df, f"spreading {self.header} from {df.df_name} into detections"
            ):
                if value is None:
                    self._drop_column(df, self.header)
                    return

                dets = df.locs.detections
                if self.header in dets.columns:
                    raise ValueError(
                        f"Cannot set {self.col} on {df.df_name} dataframe as it already exists "
                        f"in detections and cannot be spread."
                    )

                df[self.header] = self._coerce_array(value)
                dets[self.header] = self._map_child_to_detections(df, self.header)

        setattr(DataFrame, self.col, merged_col)

    def _install_index_property_on_main(self):
        """Install index property on MainDataFrame."""
        if hasattr(MainDataFrame, self.col):
            raise SyntaxError(f"{self.col} cannot be defined twice in MainDataFrame")

        @property
        def index_col(dets):
            """Return index col."""
            if self.header in dets.columns:
                return dets[self.header].to_numpy()

            newcol = self.func(dets)
            if newcol is None:
                return None

            if isinstance(newcol, str):
                newcol = getattr(dets, newcol)

            setattr(dets, self.col, newcol)
            return getattr(dets, self.col)

        @index_col.setter
        def index_col(dets, value):
            """Set index col."""
            if value is None:
                self._drop_column(dets, self.header)
                if hasattr(dets, "df_dict"):
                    dets.df_dict.pop(self.df_name, None)
                return

            dets[self.header] = self._coerce_array(value)

        setattr(MainDataFrame, self.col, index_col)

    def _install_locsreceiver_dataframe_access(self):
        """Install lazy dataframe access on LocsReceiver."""
        if hasattr(LocsReceiver, self.df_name):
            raise SyntaxError(f"{self.df_name} cannot be defined twice in LocsReceiver")

        @property
        def get_df(locs):
            """Return get df."""
            if self.df_name not in locs.df_dict:
                if getattr(locs.detections, self.col, None) is None:
                    return None

                from smlmlp import dataframes

                locs.df_dict[self.df_name] = dataframes[self.df_name](locs)

            return locs.df_dict[self.df_name]

        setattr(LocsReceiver, self.df_name, get_df)

        len_name = f"n{self.df_name}"
        if hasattr(LocsReceiver, len_name):
            raise SyntaxError(f"{len_name} cannot be defined twice in LocsReceiver")

        @property
        def len_df(locs):
            """Return len df."""
            df = getattr(locs, self.df_name)
            return 0 if df is None else len(df)

        setattr(LocsReceiver, len_name, len_df)

    def _install_main_dataframe_spread_property(self):
        """Install spread property on MainDataFrame."""
        if hasattr(MainDataFrame, self.col):
            raise SyntaxError(f"{self.col} cannot be defined twice in MainDataFrame")

        @property
        def spread_col(dets):
            """Return spread col."""
            with self._printer_context(
                dets, f"spreading {self.col} from {self.df_name} into detections"
            ):
                if self.header not in dets.columns:
                    df = getattr(dets.locs, self.df_name)
                    if df is None:
                        return None

                    if self.header not in df.columns:
                        if not self._ensure_column_materialized(
                            df,
                            self.col,
                            self.header,
                            allow_none=True,
                        ):
                            return None

                    spread = self._map_child_to_detections(
                        df,
                        self.header,
                        allow_none=True,
                    )
                    if spread is None:
                        return None

                    dets[self.header] = spread

                return dets[self.header].to_numpy()

        @spread_col.setter
        def spread_col(dets, value):
            """Set spread col."""
            with self._printer_context(
                dets, f"merging {self.header} from detections into {self.df_name}"
            ):
                if value is None:
                    self._drop_column(dets, self.header)
                    return

                df = getattr(dets.locs, self.df_name)
                if df is None:
                    raise ValueError(
                        f"{self.cls.index_header} not in detections and cannot be used "
                        f"to merge {self.header} into {self.df_name}"
                    )

                if self.header in df.columns:
                    raise ValueError(
                        f"Cannot set {self.header} on detections dataframe as it already exists "
                        f"in its normal parent {self.df_name} dataframe and cannot be merged."
                    )

                dets[self.header] = self._coerce_array(value)

                parent = getattr(df.locs, df.parent_name)

                if self.header not in parent.columns:
                    self._ensure_column_materialized(parent, self.col, self.header)

                if df.index_header not in parent.columns:
                    self._ensure_column_materialized(parent, df.index_col, df.index_header)

                df[self.header] = self._aggregate_parent_to_child(
                    parent=parent,
                    child=df,
                    value_header=self.header,
                )

        setattr(MainDataFrame, self.col, spread_col)

    # Descriptor API

    def __get__(self, instance, cls):
        """Implement __get__."""
        if instance is None:
            return self

        if self.index:
            return instance.index.to_numpy()

        if self.header in instance.columns:
            return instance[self.header].to_numpy()

        newcol = self.func(instance)
        if newcol is None:
            return None

        if isinstance(newcol, str):
            return getattr(instance, newcol)

        setattr(instance, self.col, newcol)
        return getattr(instance, self.col)

    def __set__(self, instance, value):
        """Implement __set__."""
        if self.index:
            raise SyntaxError(
                "Setting Dataframe index is not possible, you need to define the index in locs.detections"
            )

        if value is None:
            self._drop_column(instance, self.header)
            return

        instance[self.header] = self._coerce_array(value)
