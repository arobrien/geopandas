from __future__ import absolute_import

from shapely.geometry import Point

import unittest
import tempfile
import pytest
from geopandas import GeoDataFrame, read_file
from geopandas import overlay
from geopandas import datasets

# Load qgis overlays
qgispath = datasets._module_path+'/qgis_overlay/'
outputpath = qgispath + 'out/'

union_qgis = read_file(qgispath+'union_qgis.shp')
diff_qgis = read_file(qgispath+'diff_qgis.shp')
symdiff_qgis = read_file(qgispath+'symdiff_qgis.shp')
intersect_qgis = read_file(qgispath+'intersect_qgis.shp')
# ident_qgis = union_qgis.copy()
ident_qgis = union_qgis[union_qgis.BoroCode.isnull()==False].copy()
# Eliminate observations without geometries (issue from QGIS)
union_qgis = union_qgis[union_qgis.is_valid]
diff_qgis = diff_qgis[diff_qgis.is_valid]
symdiff_qgis = symdiff_qgis[symdiff_qgis.is_valid]
intersect_qgis = intersect_qgis[intersect_qgis.is_valid]
ident_qgis = ident_qgis[ident_qgis.is_valid]

# Order GeoDataFrames
cols = ['BoroCode', 'BoroName', 'Shape_Leng', 'Shape_Area', 'value1', 'value2']
union_qgis.sort_values(cols, inplace=True)
symdiff_qgis.sort_values(cols, inplace=True)
intersect_qgis.sort_values(cols, inplace=True)
ident_qgis.sort_values(cols, inplace=True)
diff_qgis.sort_values(cols[:-2], inplace=True)

# Reset indexes
union_qgis.reset_index(inplace=True, drop=True)
symdiff_qgis.reset_index(inplace=True, drop=True)
intersect_qgis.reset_index(inplace=True, drop=True)
ident_qgis.reset_index(inplace=True, drop=True)
diff_qgis.reset_index(inplace=True, drop=True)

class TestDataFrame(unittest.TestCase):
    def setUp(self):
        # Create original data again
        N = 10
        nybb_filename = datasets.get_path('nybb')
        print(nybb_filename)
        self.polydf = read_file(nybb_filename)
        self.tempdir = tempfile.mkdtemp()
        self.crs = {'init': 'epsg:4326'}
        b = [int(x) for x in self.polydf.total_bounds]
        self.polydf2 = GeoDataFrame([
            {'geometry' : Point(x, y).buffer(10000), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)
        self.pointdf = GeoDataFrame([
            {'geometry' : Point(x, y), 'value1': x + y, 'value2': x - y}
            for x, y in zip(range(b[0], b[2], int((b[2]-b[0])/N)),
                            range(b[1], b[3], int((b[3]-b[1])/N)))], crs=self.polydf.crs)

        # TODO this appears to be necessary;
        # why is the sindex not generated automatically?
        #self.polydf2._generate_sindex()

        self.union_shape = union_qgis.shape

    def test_union(self):
        df = overlay(self.polydf, self.polydf2, how="union")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, union_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEqual((df.area/union_qgis.area).mean(),1)
        self.assertEqual((df.boundary.length/union_qgis.boundary.length).mean(),1)

    def test_intersection(self):
        df = overlay(self.polydf, self.polydf2, how="intersection")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, intersect_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEqual((df.area/intersect_qgis.area).mean(),1)
        self.assertEqual((df.boundary.length/intersect_qgis.boundary.length).mean(),1)

    def test_identity(self):
        df = overlay(self.polydf, self.polydf2, how="identity")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        print(df.shape)
        print(ident_qgis.shape)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, ident_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEqual((df.area/ident_qgis.area).mean(),1)
        self.assertEqual((df.boundary.length/ident_qgis.boundary.length).mean(),1)
        self.assertEqual(1,2)

    def test_symmetric_difference(self):
        df = overlay(self.polydf, self.polydf2, how="symmetric_difference")
        df.sort_values(cols, inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, symdiff_qgis.shape)
        self.assertTrue('value1' in df.columns and 'Shape_Area' in df.columns)
        self.assertEqual((df.area/symdiff_qgis.area).mean(),1)
        self.assertEqual((df.boundary.length/symdiff_qgis.boundary.length).mean(),1)

    def test_difference(self):
        df = overlay(self.polydf, self.polydf2, how="difference")
        df.sort_values(cols[:-2], inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.assertTrue(type(df) is GeoDataFrame)
        self.assertEqual(df.shape, diff_qgis.shape)
        self.assertTrue('value1' not in df.columns and 'Shape_Area' in df.columns)
        self.assertEqual((df.area/diff_qgis.area).mean(),1)
        self.assertEqual((df.boundary.length/diff_qgis.boundary.length).mean(),1)

    def test_bad_how(self):
        with pytest.raises(ValueError):
            overlay(self.polydf, self.polydf, how="spandex")

    def test_nonpoly(self):
        with pytest.raises(TypeError):
            overlay(self.pointdf, self.polydf, how="union")

    def test_duplicate_column_name(self):
        polydf2r = self.polydf2.rename(columns={'value2': 'Shape_Area'})
        df = overlay(self.polydf, polydf2r, how="union")
        self.assertTrue('Shape_Area_2' in df.columns and 'Shape_Area_1' in df.columns)

    def test_geometry_not_named_geometry(self):
        # Issue #306
        # Add points and flip names
        polydf3 = self.polydf.copy()
        polydf3 = polydf3.rename(columns={'geometry': 'polygons'})
        polydf3 = polydf3.set_geometry('polygons')
        polydf3['geometry'] = self.pointdf.geometry.loc[0:4]
        assert polydf3.geometry.name == 'polygons'

        df = overlay(polydf3, self.polydf2, how="union")
        assert type(df) is GeoDataFrame

        df2 = overlay(self.polydf, self.polydf2, how="union")
        assert df.geom_almost_equals(df2).all()

    def test_geoseries_warning(self):
        # Issue #305
        def f():
            overlay(self.polydf, self.polydf2.geometry, how="union")
