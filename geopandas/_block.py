from __future__ import absolute_import, division, print_function

import numpy as np

from pandas.core.internals import Block, NonConsolidatableMixIn
from pandas.core.common import is_null_slice
from pandas.core.dtypes.inference import is_list_like
from pandas.core.indexing import length_of_indexer
from shapely.geometry.base import geom_factory, BaseGeometry

from .vectorized import GeometryArray, to_shapely, concat


class GeometryBlock(NonConsolidatableMixIn, Block):
    """ Pandas Geometry block with pointers to C GEOS objects """
    __slots__ = ()

    @property
    def _holder(self):
        return GeometryArray

    def __init__(self, values, placement, ndim=2, **kwargs):

        if not isinstance(values, self._holder):
            raise TypeError("values must be a GeometryArray object")

        super(GeometryBlock, self).__init__(values, placement=placement,
                                            ndim=ndim, **kwargs)

    @property
    def _box_func(self):
        # TODO does not seems to be used at the moment (from the examples) ?
        print("I am boxed")
        return geom_factory

    # @property
    # def _na_value(self):
    #     return None
    #
    # @property
    # def fill_value(self):
    #     return tslib.iNaT

    # TODO
    # def copy(self, deep=True, mgr=None):
    #     """ copy constructor """
    #     values = self.values
    #     if deep:
    #         values = values.copy(deep=True)
    #     return self.make_block_same_class(values)

    def external_values(self):
        """ we internally represent the data as a DatetimeIndex, but for
        external compat with ndarray, export as a ndarray of Timestamps
        """
        #return np.asarray(self.values)
        print("I am densified (external_values, {} elements)".format(len(self)))
        return self.values.to_dense()

    def formatting_values(self, dtype=None):
        """ return an internal format, currently just the ndarray
        this should be the pure internal API format
        """
        return self.to_dense()

    def to_dense(self):
        print("I am densified ({} elements)".format(len(self)))
        return self.values.to_dense().view()

    def _getitem(self, indexer):
        values = self.values[indexer]
        return GeometryBlock(values, placement=slice(0, len(values), 1),
                             ndim=1)

    def setitem(self, indexer, value, mgr=None):
        # Overrides Block.setitem() in pandas/core/internals.py
        # The majority of this code is copied from there.
        #
        # the problem with the overridden method are:
        # - Shapely shapes imitate arrays <- pandas should implement special protection logic like they do for strings and arrays in is_list_like()
        # - np.array(geoms) will (except for Polygons) create an array of the coordinates <- pandas should fix this by using np.array(value,dtype='object')
        # - make_block() returns a generic ObjectBlock

        values = self.values

        # check whether we have a geometry
        if isinstance(value, BaseGeometry):
            # dtype='object' protects Shapely objects from coordinate extraction
            arr_value = np.array([value],dtype='object',copy=False)
            single_value = True
        elif is_list_like(value):
            l = len(value)
            for i in range(l):
                if not isinstance(value[i], BaseGeometry):
                    raise TypeError("Element {} of values is not a geometry".format(value[i]))
            # dtype='object' protects Shapely objects from coordinate extraction
            arr_value = np.array(value,dtype='object',copy=False)
            single_value = False
        else:
            raise TypeError('Can only store Shapely geometry, not {}'.format(value))

        l = len(values)

        # length checking
        # boolean with truth values == len of the value is ok too
        if isinstance(indexer, (np.ndarray, list)):
            if not single_value and len(indexer) != len(value):
                if not (isinstance(indexer, np.ndarray) and
                        indexer.dtype == np.bool_ and
                        len(indexer[indexer]) == len(value)):
                    raise ValueError("cannot set using a list-like indexer "
                                     "with a different length than the value")

        # slice
        elif isinstance(indexer, slice):

            if not single_value and l:
                if len(value) != length_of_indexer(indexer, values):
                    raise ValueError("cannot set using a slice indexer with a "
                                     "different length than the value")

        def _is_scalar_indexer(indexer):
            # return True if we are all scalar indexers

            if arr_value.ndim == 1:
                if not isinstance(indexer, tuple):
                    indexer = tuple([indexer])
                    return any(isinstance(idx, np.ndarray) and len(idx) == 0
                               for idx in indexer)
            return False

        def _is_empty_indexer(indexer):
            # return a boolean if we have an empty indexer

            if is_list_like(indexer) and not len(indexer):
                return True
            if arr_value.ndim == 1:
                if not isinstance(indexer, tuple):
                    indexer = tuple([indexer])
                return any(isinstance(idx, np.ndarray) and len(idx) == 0
                           for idx in indexer)
            return False

        # empty indexers
        # 8669 (empty)
        if _is_empty_indexer(indexer):
            pass

        # setting a single element for each dim and with a rhs that could
        # be say a list
        # GH 6043
        elif _is_scalar_indexer(indexer):
            values[indexer] = value

        # if we are an exact match (ex-broadcasting),
        # then use the resultant dtype
        elif (len(arr_value.shape) and
              arr_value.shape[0] == values.shape[0] and
              np.prod(arr_value.shape) == np.prod(values.shape)):
            values[indexer] = value
            try:
                values = values.astype(arr_value.dtype)
            except ValueError:
                pass

        # set
        else:
            values[indexer] = value

        return GeometryBlock(values, placement=slice(0, len(values), 1),
                             ndim=1)

    # TODO is this needed?
    # def get_values(self, dtype=None):
    #     """
    #     return object dtype as boxed values, as shapely objects
    #     """
    #     if is_object_dtype(dtype):
    #         return lib.map_infer(self.values.ravel(),
    #                              self._box_func).reshape(self.values.shape)
    #     return self.values

    def to_native_types(self, slicer=None, na_rep=None, date_format=None,
                        quoting=None, **kwargs):
        """ convert to our native types format, slicing if desired """

        values = self.values
        if slicer is not None:
            values = values[slicer]

        values = to_shapely(values.data)

        return np.atleast_2d(values)

    # TODO needed for what?
    def _can_hold_element(self, element):
        # if is_list_like(element):
        #     element = np.array(element)
        #     return element.dtype == _NS_DTYPE or element.dtype == np.int64
        return isinstance(element, BaseGeometry)

    def _slice(self, slicer):
        """ return a slice of my values """
        if isinstance(slicer, tuple):
            col, loc = slicer
            if not is_null_slice(col) and col != 0:
                raise IndexError("{0} only contains one item".format(self))
            return self.values[loc]
        return self.values[slicer]

    def take_nd(self, indexer, axis=0, new_mgr_locs=None, fill_tuple=None):
        """
        Take values according to indexer and return them as a block.bb
        """
        if fill_tuple is None:
            fill_value = None
        else:
            fill_value = fill_tuple[0]

        # axis doesn't matter; we are really a single-dim object
        # but are passed the axis depending on the calling routing
        # if its REALLY axis 0, then this will be a reindex and not a take

        # TODO implement take_nd on GeometryArray
        # new_values = self.values.take_nd(indexer, fill_value=fill_value)
        new_values = self.values.take(indexer)

        # if we are a 1-dim object, then always place at 0
        if self.ndim == 1:
            new_mgr_locs = [0]
        else:
            if new_mgr_locs is None:
                new_mgr_locs = self.mgr_locs

        return self.make_block_same_class(new_values, new_mgr_locs)

    def eval(self, func, other, raise_on_error=True, try_cast=False,
             mgr=None):
        if func.__name__ == 'eq':
            super(GeometryBlock, self).eval(
                func, other, raise_on_error=raise_on_error, try_cast=try_cast,
                mgr=mgr)
        raise TypeError("{} not supported on geometry blocks".format(func.__name__))


    def _astype(self, dtype, copy=False, errors='raise', values=None,
                klass=None, mgr=None):
        """
        Coerce to the new type (if copy=True, return a new copy)
        raise on an except if raise == True
        """

        if dtype == np.object_:
            values = self.to_dense()
        elif dtype == str:
            values = np.array(list(map(str, self.to_dense())))
        else:
            if errors == 'raise':
                raise TypeError('cannot astype geometries')
            else:
                values = self.to_dense()

        if copy:
            values = values.copy()

        return self.make_block(values)

    # def should_store(self, value):
    #     return (issubclass(value.dtype.type, np.uint64)
    #             and value.dtype == self.dtype)

    def set(self, locs, values, check=False):
        """
        Modify Block in-place with new item value

        Returns
        -------
        None
        """
        if values.dtype != self.dtype:
            # Workaround for numpy 1.6 bug
            if isinstance(values, BaseGeometry):
                values = values.__geom__
            else:
                raise ValueError()

            self.values[locs] = values

    def concat_same_type(self, to_concat, placement=None):
        """
        Concatenate list of single blocks of the same type.
        """
        values = concat([blk.values for blk in to_concat])
        return self.make_block_same_class(
            values, placement=placement or slice(0, len(values), 1))
