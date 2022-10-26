# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:20:33 2020
@author: santi
"""
import zipfile
import os
import pandas as pd
import partridge as ptg
import geopandas as gpd
import utm
from shapely.ops import nearest_points
from shapely.geometry import Point, LineString, MultiLineString, MultiPoint
from shapely.ops import split
import math
import re


# os.system('apt install libspatialindex-dev')
# os.system('pip install rtree')

import warnings
warnings.filterwarnings("ignore")


def import_gtfs(gtfs_path):
    # Partridge to read the feed
    # service_ids = pd.read_csv(gtfs_path + '/trips.txt')['service_id'].unique()
    # service_ids = frozenset(tuple(service_ids))
    # if busiest_date:
    #     service_ids = ptg.read_busiest_date(gtfs_path)[1]
    # else:
    #     with zipfile.ZipFile(gtfs_path) as myzip:
    #         myzip.extract("trips.txt")
    #     service_ids = pd.read_csv('trips.txt')['service_id'].unique()
    #     service_ids = frozenset(tuple(service_ids))
    #     os.remove('trips.txt')

    # Leaving only the option for the busiest service_id
    service_ids = ptg.read_busiest_date(gtfs_path)[1]

    view = {'trips.txt': {'service_id': service_ids}}

    feed = ptg.load_geo_feed(gtfs_path, view)

    routes = feed.routes
    trips = feed.trips
    stop_times = feed.stop_times
    stops = feed.stops
    shapes = feed.shapes

    # Get routes info in trips
    # The GTFS feed might be missing some of the keys, e.g. direction_id or shape_id.
    # To allow processing incomplete GTFS data, we must reindex instead:
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike
    # This will add NaN for any missing columns.
    trips = pd.merge(trips, routes, how='left').reindex(
        columns=[
            'trip_id', 'route_id', 'service_id', 'direction_id', 'shape_id'])

    # Get trips, routes and stops info in stop_times
    stop_times = pd.merge(stop_times, trips, how='left')
    stop_times = pd.merge(stop_times, stops, how='left')

    # stop_times needs to be geodataframe if we want to do geometry operations
    stop_times = gpd.GeoDataFrame(stop_times, geometry='geometry')

    # direction_id is optional, as it is not needed to determine route shapes
    # However, if direction_id is NaN, pivot_table will return an empty DataFrame.
    # Therefore, use a sensible default if direction id is not known.
    # Some gtfs feeds only contain direction_id 0, use that as default
    stop_times['direction_id'] = stop_times['direction_id'].fillna(0)

    return routes, stops, stop_times, trips, shapes


def cut_gtfs(stop_times, stops, shapes):
    # Get the right epsg code for later convertions
    shapes.crs = {'init': 'epsg:4326'}

    lat = shapes.geometry.iloc[0].coords[0][1]
    lon = shapes.geometry.iloc[0].coords[0][0]

    zone = utm.from_latlon(lat, lon)

    if lat < 0:
        epsg = 32700 + zone[2]
    else:
        epsg = 32600 + zone[2]

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # --------------------- FIND THE CLOSEST POINT TO EACH LINE --------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------ 

    # Data frame with stop sequence for route and direction
    sseq = stop_times.drop_duplicates(subset=['stop_id','stop_name', 'stop_sequence', 'shape_id'])[['route_id','direction_id','stop_id','stop_name', 'stop_sequence', 'shape_id']]

    # Data frames with the number of stops for each route and direction and shape_id
    route_shapes = sseq.pivot_table('stop_id',
                               index = ['route_id', 'direction_id', 'shape_id'],
                               aggfunc='count').reset_index()
    route_shapes.columns = ['route_id','direction_id', 'shape_id', 'stops_count']

    # List of shape_ids
    shape_id_list = shapes.shape_id.unique()

    # Create a DataFrame with the pair (stop, nearest_point) for each shape_id
    def find_shape_closest_points(shape_id):
        #shape_id = row.shape_id
        route_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'route_id'].values[0]
        direction_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'direction_id'].values[0]

        # Look for the shape
        shape = shapes.loc[shapes.shape_id == shape_id,'geometry'].values[0]


        # Look for the stop_ids of this shape
        route_stop_ids = sseq.loc[(sseq['route_id'] == route_id) 
                                  & (sseq['direction_id'] == direction_id)
                                  &(sseq['shape_id'] == shape_id)]

        # Look for the geometry of these stops
        # merged = pd.merge(route_stop_ids, stops, how='left')
        # route_stop_geom = merged.geometry
        route_stop_geom = pd.merge(route_stop_ids, stops, how='left').geometry

        # Look for the nearest points of these stops that are in the shape
        points_in_shape = route_stop_geom.apply(lambda x: nearest_points(x, shape))

        d = dict(shape_id=shape_id, points=list(points_in_shape))

        return d

    shape_closest_points = [find_shape_closest_points(s) for s in shape_id_list]

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # --------------------- CREATE LINES THAT CUT THE SHAPE ------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------

    shape_trans_lines = pd.DataFrame()
    # First we define a function that will help us create the line to intersect the shape

    # ---------------- THIS IS THE VALUE YOU SHOULD CHANGE IF THE CUTTING GEOMETRY AND ---
    # ---------------- THE LINE INTERSECT -------------------------------------------------
    offset = 0.0001

    def create_line(row):
        # Formula to make the line longer
        # a = (y1-b)/x1
        # b = (y2-x2/x1*y1)/(1-x2/x1)
        if row[0] == row[1]:
            x1 = row[0].x - offset
            y1 = row[0].y - offset

            x2 = row[0].x 
            y2 = row[0].y

            x3 = row[0].x + offset
            y3 = row[0].y + offset

        else:   
            x1 = row[0].x
            y1 = row[0].y

            x2 = row[1].x
            y2 = row[1].y

            # If x2==x1 it will give the error "ZeroDivisionError"
            if float(x2) != float(x1):
                b = (y2-x2/x1*y1)/(1-x2/x1)
                a = (y1-b)/x1

                if x2 - x1 < 0: # We should create an "if" to check if we need to do -1 or +1 depending on x2-x1
                    x3 = x2 - 3*(x1 - x2)#offset
                else:
                    x3 = x2 + 3*(x2 - x1)#offset

                y3 = a*x3 + b

            else:
                x3 = x2
                b = 0
                a = 0

                if y2-y1 < 0:
                    #y3 = y2 - offset/5
                    y3 = y2 - 3*(y1-y2) #offset/10000000
                else: 
                    #y3 = y2 + offset/5
                    y3 = y2 + 3*(y2-y1) #offset/10000000

        trans = LineString([Point(x1,y1), Point(x2,y2), Point(x3, y3)])
        return trans

    # For each shape we need to create transversal lines and separete the shape in segments    
    def find_shape_trans_lines(shape_closest_points):
        # Choose the shape
        shape_id = shape_closest_points['shape_id']

        # Choose the pair (stop, nearest point to shape) to create the line
        scp = shape_closest_points['points']

        lines = [create_line(p) for p in scp]
    #    scp.apply(create_line)

        d = dict(shape_id=shape_id, trans_lines=lines)

        return d

    shape_trans_lines = [find_shape_trans_lines(shape_closest_points[i]) for i in range(0, len(shape_closest_points))]

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------ CUT THE SHAPES --------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # Set the tolerance of the cuts
    tolerance = 0.0001

    loops_route_id = []
    loops_direction_id = []
    loops_shape_id = []

    def cut_shapes_(shape_trans_lines, shape_closest_points):
        shape_id = shape_trans_lines['shape_id']
        route_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'route_id'].values[0]
        direction_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'direction_id'].values[0]

        # Check if the line is simple (ie, doesn't intersect itself)
        line = shapes.loc[shapes.shape_id == shape_id, 'geometry'].values[0]
        if line.is_simple:
            # Split the shape in different segments
            trans_lines = shape_trans_lines['trans_lines']

            df = sseq.loc[(sseq['route_id'] == route_id) 
                          & (sseq['direction_id'] == direction_id)
                          & (sseq['shape_id'] == shape_id)].reset_index()


            #df['segment'] = ''

            d = dict(shape_id = shape_id,route_id=route_id, direction_id=direction_id, stop_id = list(df.stop_id)[:-1], stop_sequence=list(df.stop_sequence)[:-1])

            if len(trans_lines) == 2:
                # In case there is a line with only two stops
                d['segment'] = [line]
                return d

            else:
                # trans_lines_all = MultiLineString(list(trans_lines.values))
                # trans_lines_cut = MultiLineString(list(trans_lines.values)[1:-1])

                # # Split the shape in different segments, cut by the linestrings created before
                # # The result is a geometry collection with the segments of the route
                # result = split(line, trans_lines_cut)
                try:
                    trans_lines_all = MultiLineString(trans_lines)
                    trans_lines_cut = MultiLineString(trans_lines[1:-1])

                    # Split the shape in different segments, cut by the linestrings created before
                    # The result is a geometry collection with the segments of the route
                    result = split(line, trans_lines_cut)
                except ValueError:
                    # If the cut points are on the line then try to cut with the points instead of lines
                    test = shape_closest_points['points']
                    cut_points = [test[i][1] for i in range(len(test))]
                    cut_points = MultiPoint(cut_points[1:-1])
                    result = split(line, cut_points)

                if len(result)==len(trans_lines_all)-1:
                    d['segment'] = [s for s in result]

                    return d
                else:
                    loops_route_id.append(route_id)
                    loops_direction_id.append(direction_id)
                    loops_shape_id.append(shape_id) 
        else:
            loops_route_id.append(route_id)
            loops_direction_id.append(direction_id)
            loops_shape_id.append(shape_id)

    segments = [cut_shapes_(shape_trans_lines[i], shape_closest_points[i])  for i in range(0, len(shape_trans_lines))]

    # Remove None values
    segments = [i for i in segments if i] 

    loops = pd.DataFrame()
    loops['route_id'] = loops_route_id
    loops['direction_id'] = loops_direction_id
    loops['shape_id'] = loops_shape_id

    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------- CUT THE SHAPES WITH LOOPS --------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------

    # Manage the lines with loops
    shapes_loop = shapes.loc[shapes.shape_id.isin(loops_shape_id)]

    aux = pd.DataFrame.from_dict(shape_trans_lines)
    trans_loop = aux.loc[aux.shape_id.isin(loops_shape_id)]

    aux = pd.DataFrame.from_dict(shape_closest_points)
    cut_points_loop = aux.loc[aux.shape_id.isin(loops_shape_id)]

    # Separate the shapes according to possible exceptions
    trans_loop['n_segments'] = trans_loop['trans_lines'].map(len)
    run_shapes_no_middle = False
    run_shapes_one_seg = False

    # Exception 1: Only three stops --> one cut point, two segments
    # If there's only one cut_point this will make the
    # script skip the "Middle segments" part
    # (with only one cut point there are only two segments)

    shapes_no_middle = shapes.loc[shapes.shape_id.isin(trans_loop.loc[trans_loop['n_segments'] ==3, 'shape_id'].unique())].reset_index()

    if len(shapes_no_middle) > 0:
        run_shapes_no_middle = True

    # Exception 2: Only two stops --> no cut points, one segments
    shapes_one_seg = shapes.loc[shapes.shape_id.isin(trans_loop.loc[trans_loop['n_segments'] ==2, 'shape_id'].unique())].reset_index()

    if len(shapes_one_seg) > 0 :
        run_shapes_one_seg = True

    # The rest of the shapes
    shapes_ok = shapes.loc[shapes.shape_id.isin(trans_loop.loc[trans_loop['n_segments'] >3, 'shape_id'].unique())].reset_index()

    def add_points(row, add_p, cut_points_gdf):
        # Calculate the min distance between the stops that intersect this segment
        index_track_ = row.name
        p = cut_points_gdf.loc[cut_points_gdf.index.isin(add_p.loc[add_p.index_track_==index_track_, 'index_cut'])]
        p.crs={'init':'epsg:4326'}

        seg = [LineString([p.geometry.values[i], p.geometry.values[i+1]]) for i in range(0,len(p)-1)]
        seg = gpd.GeoSeries(seg)
        seg.crs={'init':'epsg:4326'}
        dist = seg.to_crs(epsg).length.min() - 5


        gse = gpd.GeoSeries(row.geometry, index=[row.distance_m])
        gse.crs = {'init':'epsg:4326'}
        gse = gse.to_crs(epsg)

        length = gse.index[0]
        start = gse.values[0].coords[0]
        end = gse.values[0].coords[-1]

        num_vert = int(length/dist)

        new_points = [start] + [gse.values[0].interpolate(dist*n) for n in list(range(1, num_vert+1))] + [end]
        new_points = [Point(p) for p in new_points]
        new_line = LineString(new_points)

        check = gpd.GeoSeries([new_line])
        check.crs = {'init':'epsg:{}'.format(epsg)}
        check = check.to_crs(epsg=4326)
        return check[0]

    # Loop lines with more than three stops
    def cut_loops_shapes_ok(shape_id):
        # Set the ids
        route_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'route_id'].values[0]
        direction_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'direction_id'].values[0]

        df = sseq.loc[(sseq['route_id'] == route_id) 
                      & (sseq['direction_id'] == direction_id)
                      & (sseq['shape_id'] == shape_id)].reset_index()

        d = dict(shape_id = shape_id,route_id=route_id, direction_id=direction_id, stop_id = list(df.stop_id)[:-1], stop_sequence=list(df.stop_sequence)[:-1])
        #d = dict(shape_id = shape_id,route_id=route_id, direction_id=direction_id, stop_id = list(df.stop_id), stop_sequence=list(df.stop_sequence))

        # All the necessary information to split the line
        # 1- line to be cut
        # 2- transversal lines to cut
        # 3- closest point on the line

        line = shapes_ok.loc[shapes_ok.shape_id == shape_id, 'geometry'].values[0]                       
        cut_lines = trans_loop.loc[trans_loop.shape_id==shape_id,'trans_lines'].values[0][1:-1] 
        cut_points = [x[1] for x in cut_points_loop.loc[cut_points_loop.shape_id==shape_id,'points'].values[0][1:-1]]

        cut_gdf = gpd.GeoDataFrame(data=list(range(len(cut_lines))), geometry=cut_lines)
        cut_points_gdf = gpd.GeoDataFrame(data=list(range(len(cut_points))), geometry=cut_points)

        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        # Make sure the shapes has a point every 100m
        # Create a GeoDataFrame with two point segments of the shape and its distance in meters
        shape = line.coords
        # Create two point segments for the shape
        track_l = gpd.GeoSeries([LineString([shape[i], shape[i+1]]) for i in range(0, len(shape)-1)])
        track_l.crs={'init':'epsg:4326'}
        #Calculate the length of each two point segment in meters
        track_dist = track_l.to_crs(epsg=epsg).length
        # Create the dataframe
        track_l_gdf = gpd.GeoDataFrame(data=dict(distance_m = track_dist), geometry = track_l)

        # Check where stops are closer than points of the track
        # To do that we intersect each segment between two segments of the track with our cut lines
        how_many = gpd.sjoin(track_l_gdf, cut_gdf, how='left', op='intersects', lsuffix='left', rsuffix='right').reset_index()
        how_many.rename(columns=dict(index='index_track_', index_right = 'index_cut'), inplace=True)

        # The filter those that were intersected by more than one cut line
        how_manyp = how_many.pivot_table('geometry', index='index_track_', aggfunc='count').reset_index()
        how_manyp = how_manyp.loc[how_manyp.geometry>1]

        add_p = how_many.loc[how_many.index_track_.isin(how_manyp.index_track_.unique())]

        # Add intermediate points for segments with length > 100m
        track_l_gdf.loc[track_l_gdf.index.isin(how_manyp.index_track_.unique()), 'geometry'] = track_l_gdf.loc[track_l_gdf.index.isin(how_manyp.index_track_.unique())] .apply(lambda x: add_points(x, add_p, cut_points_gdf), axis=1)

        #track_l_gdf.loc[track_l_gdf.distance_m>dist, 'geometry'] = track_l_gdf.loc[track_l_gdf.distance_m>dist].apply(lambda x: add_points(x, dist), axis=1)

        # Take the points and create the LineString again
        t = [list(g.coords)[:-1] for g in track_l_gdf.geometry]
        flat_list = [item for sublist in t for item in sublist] + [track_l_gdf.geometry.tail(1).values[0].coords[-1]]

        line = LineString(flat_list)    

        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        # First segment
        # We will use i to identify were the next segment should start
        for i in range(2, len(line.coords)):
            segment = LineString(line.coords[0:i])
            if segment.intersects(cut_lines[0]):
                points_to_stop = line.coords[0:i-1] + list(cut_points[0].coords)
                segment = LineString(points_to_stop)

                # Save the position of the point that makes it to the intersection
                #last_point = i
                last_point = i-1
                d['segment'] = [segment]
                #df.loc[0, 'segment'] = segment                       # assign the linestring to that segment

                break

        # Middle segments
        for l in range(1, len(cut_lines)):
            nearest_point = list(cut_points[l-1].coords)            # segments always start in the one of the cut points
            start_iterator = last_point + 1                         # start from the last point found in the previous segment

            for i in range(start_iterator, len(line.coords)+1):
                points_to_stop = nearest_point + line.coords[last_point:i]  # keep adding points to extend the line
                segment = LineString(points_to_stop)

                if segment.intersects(cut_lines[l]):                        
                    # if the line intersects with the cut line, define the segment
                    # the segment goes from one cut point to the next one
                    points_to_stop = nearest_point + line.coords[last_point:i-1] + list(cut_points[l].coords)
                    segment = LineString(points_to_stop)

                    # Save the position of the point that makes it to the intersection
                    last_point = i-1
                    d['segment'] = d['segment'] + [segment]
                    break 

                if i==(len(line.coords)):
                    points_to_stop = nearest_point + list(cut_points[l].coords)
                    segment = LineString(points_to_stop)
                    d['segment'] = d['segment'] + [segment]

        # Last segment
        # We start at the last cut point and go all the way to the end
        nearest_point = list(cut_points[l].coords)
        points_to_stop = nearest_point + line.coords[last_point:len(line.coords)]
        segment = LineString(points_to_stop)

        d['segment'] = d['segment'] + [segment]   

        return d

    segments1 = [cut_loops_shapes_ok(s) for s in shapes_ok.shape_id.unique()]
    # Remove None values
    segments1 = [i for i in segments1 if i] 
    segments.extend(segments1)

    # Exception 1: Only three stops --> one cut point, two segments
    # If there's only one cut_point this will make the
    # script skip the "Middle segments" part
    # (with only one cut point there are only two segments)

    if run_shapes_no_middle:
        #for index, row in shapes_no_middle.iterrows():
        def cut_shapes_no_middle(shape_id):
            # Set the ids
            route_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'route_id'].values[0]
            direction_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'direction_id'].values[0]

            df = sseq.loc[(sseq['route_id'] == route_id) 
                          & (sseq['direction_id'] == direction_id)
                          & (sseq['shape_id'] == shape_id)].reset_index()

            d = dict(shape_id = shape_id, route_id=route_id, direction_id=direction_id, stop_id = list(df.stop_id)[:-1], stop_sequence=list(df.stop_sequence)[:-1])

            # All the necessary information to split the line
            # 1- line to be cut
            # 2- transversal lines to cut
            # 3- closest point on the line

            line = shapes_no_middle.loc[shapes_no_middle.shape_id == shape_id, 'geometry'].values[0]                       
            cut_lines = trans_loop.loc[trans_loop.shape_id==shape_id,'trans_lines'].values[0][1:-1] 
            cut_points = [x[1] for x in cut_points_loop.loc[cut_points_loop.shape_id==shape_id,'points'].values[0][1:-1]]

            # First segment
            # We will use i to identify were the next segment should start
            for i in range(2, len(line.coords)):
                segment = LineString(line.coords[0:i])

                if segment.intersects(cut_lines[0]):
                    points_to_stop = line.coords[0:i-1] + list(cut_points[0].coords)
                    segment = LineString(points_to_stop)

                    # Save the position of the point that makes it to the intersection
                    last_point = i
                    d['segment'] = [segment]
                    #df.loc[0, 'segment'] = segment                       # assign the linestring to that segment

                    break

            # Last segment
            # We start at the last cut point and go all the way to the end
            nearest_point = list(cut_points[0].coords)
            points_to_stop = nearest_point + line.coords[last_point-1:len(line.coords)]
            segment = LineString(points_to_stop)

            d['segment'] = d['segment'] + [segment]

            return d

        # Apply the function
        segments2 = [cut_shapes_no_middle(s) for s in shapes_no_middle.shape_id.unique()]
        # Remove None values
        segments2 = [i for i in segments2 if i] 
        segments.extend(segments2)

    # Exception 2: Only two stops --> no cut points, one segments
    if run_shapes_one_seg:
        #for index, row in shapes_one_seg.iterrows():
        def cut_shapes_one_seg(shape_id):
            # Set the ids
            route_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'route_id'].values[0]
            direction_id = route_shapes.loc[route_shapes.shape_id == shape_id, 'direction_id'].values[0]

            df = sseq.loc[(sseq['route_id'] == route_id) 
                          & (sseq['direction_id'] == direction_id)
                          & (sseq['shape_id'] == shape_id)].reset_index()

            #df['segment'] = ''
            d = dict(shape_id = shape_id,route_id=route_id, direction_id=direction_id, stop_id = list(df.stop_id)[:-1], stop_sequence=list(df.stop_sequence)[:-1])

            line = shapes_one_seg.loc[shapes_one_seg.shape_id == shape_id, 'geometry'].values[0]                       
            d['segment'] = [line]
            return d

        # Apply function
        segments3 = [cut_shapes_one_seg(s) for s in shapes_one_seg.shape_id.unique()]
        # Remove None values
        segments3 = [i for i in segments3 if i] 
        segments.extend(segments3)


    def format_shapes(s, last_id):
        df = pd.DataFrame()
        df['stop_sequence'] = s['stop_sequence']
        df['start_stop_id'] = s['stop_id']
        df['end_stop_id'] = s['stop_id'][1:] + [last_id]
        df['shape_id'] = s['shape_id']
        df['route_id'] = s['route_id']
        df['direction_id'] = s['direction_id']

        df['geometry'] = s['segment']

        return df

    df = pd.concat([format_shapes(s, sseq.loc[sseq.shape_id==s['shape_id']].tail(1).stop_id.values[0]) for s in segments])

    df = pd.merge(df, stops[['stop_id', 'stop_name']], left_on='start_stop_id', right_on='stop_id', how='left').drop('stop_id', axis=1)
    df.rename(columns=dict(stop_name='start_stop_name'), inplace=True)
    df = pd.merge(df, stops[['stop_id', 'stop_name']], left_on='end_stop_id', right_on='stop_id', how='left').drop('stop_id', axis=1)
    df.rename(columns=dict(stop_name='end_stop_name'), inplace=True)
    df['segment_id'] = df.start_stop_id + '-' + df.end_stop_id

    segments_gdf = gpd.GeoDataFrame(data = df.loc[:,['route_id','direction_id','stop_sequence','start_stop_name', 'end_stop_name', 'start_stop_id', 'end_stop_id','segment_id','shape_id']], geometry = df.geometry)

    segments_gdf.crs = {'init':'epsg:4326'}
    segments_gdf['distance_m'] = segments_gdf.geometry.to_crs(epsg=epsg).length

    return segments_gdf


def speeds_from_gtfs(
        routes, stop_times, segments_gdf,
        cutoffs=[0, 6, 9, 15, 19, 22, 24]):
    routes = routes
    stop_times = stop_times

    # Get the runtime between stops
    stop_times.sort_values(
        by=['trip_id', 'stop_sequence'], ascending=True, inplace=True)

    first_try = stop_times.loc[:, ['trip_id', 'arrival_time']]
    first_try['trip_id_next'] = first_try['trip_id'].shift(-1)
    first_try['arrival_time_next'] = first_try['arrival_time'].shift(-1)

    def runtime(row):
        if row.trip_id == row.trip_id_next:
            runtime = (row.arrival_time_next - row.arrival_time)/3600
        else:
            runtime = 0

        return runtime

    first_try['runtime_h'] = first_try.apply(runtime, axis=1)

    if len(first_try) == len(stop_times):
        stop_times['runtime_h'] = first_try['runtime_h']

    stop_times.head(2)

    # Merge stop_times with segments_gdf to get the distance
    segments_gdf['direction_id'] = segments_gdf['direction_id'].map(int)
    segments_gdf['stop_sequence'] = segments_gdf['stop_sequence'].map(int)

    speeds = pd.merge(stop_times, segments_gdf[['route_id', 'direction_id', 'start_stop_id', 'stop_sequence', 'segment_id','shape_id', 'distance_m']], 
                      left_on = ['route_id', 'direction_id', 'stop_id', 'stop_sequence', 'shape_id'], 
                      right_on = ['route_id', 'direction_id', 'start_stop_id', 'stop_sequence', 'shape_id'],
                      how = 'left').drop('start_stop_id', axis=1)

    speeds = speeds.loc[~speeds.distance_m.isnull(),
                        ['trip_id', 'route_id', 'direction_id', 'shape_id', 'segment_id',
                         'arrival_time', 'departure_time', 'stop_id','stop_name',
                         'stop_sequence', 'runtime_h', 'distance_m','geometry']
                       ]

    # Assign a time window to each row
    if max(cutoffs)<=24:
        speeds_ok = speeds.loc[speeds.departure_time < 24*3600]
        speeds_fix = speeds.loc[speeds.departure_time >= 24*3600]
        speeds_fix['departure_time'] = [d - 24*3600 for d in speeds_fix.departure_time]

        speeds = speeds_ok.append(speeds_fix)
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                l = str(w) + ':00'
            else:
                n = math.modf(w)
                l=  str(int(n[1])) + ':' + str(int(n[0]*60))
            labels = labels + [l]
    else:
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if w > 24:
                    w1 = w-24
                    l = str(w1) + ':00'
                else:
                    l = str(w) + ':00'
                labels = labels + [l]
            else:
                if w > 24:
                    w1 = w-24
                    n = math.modf(w1)
                    l = str(int(n[1])) + ':' + str(int(n[0]*60))
                else:
                    n = math.modf(w)
                    l = str(int(n[1])) + ':' + str(int(n[0]*60))
                labels = labels + [l]

    labels = [labels[i] + '-' + labels[i+1] for i in range(0, len(labels)-1)]

    speeds['departure_time'] = speeds['departure_time']/3600

    # Put each trips in the right window
    speeds['window'] = pd.cut(speeds['departure_time'], bins=cutoffs, right=False, labels=labels)
    speeds = speeds.loc[~speeds.window.isnull()]
    speeds['window'] = speeds['window'].astype(str)

    # Calculate the speed
    speeds.loc[speeds.runtime_h == 0.0, 'runtime_h'] = speeds.loc[speeds.runtime_h != 0.0, 'runtime_h'].mean()
    speeds['speed'] = round(speeds['distance_m']/1000/speeds['runtime_h'])
    speeds = speeds.loc[~speeds.speed.isnull()]

    # Calculate average speed to modify outliers
    avg_speed_route = speeds.pivot_table('speed',
                                         index=['route_id', 'direction_id','window'],
                                         aggfunc='mean').reset_index()
    avg_speed_route.rename(columns={'speed':'avg_speed_route'}, inplace=True)
    # Assign average speed to outliers
    speeds = pd.merge(speeds, avg_speed_route, how='left')
    speeds.loc[speeds.speed>120,'speed'] = speeds.loc[speeds.speed>120,'avg_speed_route']

    # Calculate max speed per segment to have a free_flow reference
    max_speed_segment = speeds.pivot_table('speed',
                                           index = ['stop_id', 'direction_id'],
                                           aggfunc='max')
    max_speed_segment.rename(columns={'speed':'max_kmh'}, inplace=True)


    # Get the average per route, direction, segment and time of day
    speeds_agg = speeds.pivot_table(['speed', 'runtime_h', 'avg_speed_route'],
                                    index=['route_id', 'direction_id', 'segment_id', 'window'],
                                    aggfunc='mean'
                                   ).reset_index()
    speeds_agg['route_id'] = speeds_agg['route_id'].map(str)
    speeds_agg['direction_id'] = speeds_agg['direction_id'].map(int)

    data = pd.merge(speeds_agg, segments_gdf, 
            left_on=['route_id', 'direction_id', 'segment_id'],
            right_on = ['route_id', 'direction_id', 'segment_id'],
            how='left').reset_index().sort_values(by = ['route_id', 'direction_id','window','stop_sequence',], ascending=True)

    data.drop(['index'], axis=1, inplace=True)

    # Route name
    routes['route_name'] = ''
    if routes.route_short_name.isnull().unique()[0]:
        routes['route_name'] = routes.route_long_name
    elif routes.route_long_name.isnull().unique()[0]:
        routes['route_name'] = routes.route_short_name
    else:
        routes['route_name'] = routes.route_short_name + ' ' + routes.route_long_name
    data = pd.merge(data, routes[['route_id', 'route_name']], left_on='route_id', right_on='route_id', how='left')

    # Get the average per segment and time of day
    # Then add it to the rest of the data

    all_lines = speeds.pivot_table(['speed', 'runtime_h', 'avg_speed_route'],
                                    index=['segment_id', 'window'],
                                    aggfunc = 'mean'
                                   ).reset_index()

    data_all_lines = pd.merge(
        all_lines,
        segments_gdf.drop_duplicates(subset=['segment_id']),
        left_on=['segment_id'],
        right_on = ['segment_id'],
        how='left').reset_index().sort_values(by = ['direction_id','window','stop_sequence'], ascending=True)

    data_all_lines.drop(['index'], axis=1, inplace=True)
    data_all_lines['route_id'] = 'ALL_LINES'
    data_all_lines['route_name'] = 'All lines'
    data_all_lines['direction_id'] = 'NA'
    data_complete = data.append(data_all_lines)

    data_complete1 = data_complete.loc[~data_complete.route_name.isnull(), :].reset_index()

    # Get the columns in the right format
    int_columns = ['speed']

    for c in int_columns:
        data_complete1[c] = data_complete1[c].apply(lambda x: round(x, 1))

    data_complete1 = data_complete1.loc[:,['route_id', 'route_name','direction_id','segment_id', 'window',
                                           'speed', 
                                           'start_stop_id', 'start_stop_name', 'end_stop_id','end_stop_name', 
                                           'distance_m','stop_sequence', 'shape_id', 'runtime_h','geometry', ]]       

    data_complete1.columns = [
        'route_id', 'route_name', 'dir_id', 'segment_id', 'window', 'speed',
        's_st_id', 's_st_name', 'e_st_id', 'e_st_name',
        'distance_m', 'stop_seq', 'shape_id', 'runtime_h', 'geometry']

    # Assign max speeds to each segment
    data_complete1 = pd.merge(
        data_complete1, max_speed_segment,
        left_on=['s_st_id', 'dir_id'], right_on=['stop_id', 'direction_id'],
        how='left')

    gdf = gpd.GeoDataFrame(
        data=data_complete1.drop('geometry', axis=1),
        geometry=data_complete1.geometry)

    gdf.loc[gdf.dir_id == 0, 'dir_id'] = 'Inbound'
    gdf.loc[gdf.dir_id == 1, 'dir_id'] = 'Outbound'

    gdf.rename(columns={'speed': 'speed_kmh'}, inplace=True)
    gdf['speed_mph'] = gdf['speed_kmh']*0.621371
    gdf['max_mph'] = gdf['max_kmh']*0.621371

    gdf = gdf.drop(['shape_id'], axis=1).drop_duplicates()

    return gdf


def add_all_lines(line_frequencies, segments_gdf):
    # Calculate sum of trips per segment with all lines
    all_lines = line_frequencies.pivot_table(
        ['ntrips'],
        index=['segment_id', 'window'],
        aggfunc='sum').reset_index()

    sort_these = ['direction_id', 'window', 'stop_sequence']

    data_all_lines = pd.merge(
        all_lines,
        segments_gdf.drop_duplicates(subset=['segment_id']),
        left_on=['segment_id'], right_on=['segment_id'],
        how='left').reset_index().sort_values(by=sort_these, ascending=True)

    data_all_lines.drop(['index'], axis=1, inplace=True)
    data_all_lines['route_id'] = 'ALL_LINES'
    data_all_lines['route_name'] = 'All lines'
    data_all_lines['direction_id'] = 'NA'
    data_complete = line_frequencies.append(data_all_lines).reset_index()

    return data_complete


def fix_departure_time(stop_times):
    """
    Reassigns departure time to trips that start after the hour 24
    """
    stop_times_ok = stop_times.loc[stop_times.departure_time < 24*3600]
    stop_times_fix = stop_times.loc[stop_times.departure_time >= 24*3600]
    stop_times_fix['departure_time'] = [
        d - 24*3600 for d in stop_times_fix.departure_time]

    stop_times = stop_times_ok.append(stop_times_fix)

    return stop_times


def label_creation(cutoffs):
    labels = []
    if max(cutoffs) <= 24:
        for w in cutoffs:
            if float(w).is_integer():
                label = str(w) + ':00'
            else:
                n = math.modf(w)
                label = str(int(n[1])) + ':' + str(int(n[0]*60))
            labels.append(label)
    else:
        labels = []
        for w in cutoffs:
            if float(w).is_integer():
                if w > 24:
                    w1 = w-24
                    label = str(w1) + ':00'
                else:
                    label = str(w) + ':00'
                labels.append(label)
            else:
                if w > 24:
                    w1 = w-24
                    n = math.modf(w1)
                    label = str(int(n[1])) + ':' + str(int(n[0]*60))
                else:
                    n = math.modf(w)
                    label = str(int(n[1])) + ':' + str(int(n[0]*60))
                labels.append(label)

    labels = [labels[i] + '-' + labels[i+1] for i in range(0, len(labels)-1)]

    return labels


def window_creation(stop_times, cutoffs):
    "Adds the time time window and labels to stop_times"
    hours = list(range(25))
    hours_labels = [str(hours[i]) + ':00' for i in range(len(hours)-1)]

    # Create the labels for the cutoffs
    if max(cutoffs) <= 24:
        stop_times = fix_departure_time(stop_times)

    labels = label_creation(cutoffs)

    # Get departure time as hour and a fraction
    departure_time = stop_times['departure_time']/3600

    # Put each trip in the right window
    stop_times['window'] = pd.cut(
        departure_time, bins=cutoffs, right=False, labels=labels)
    stop_times = stop_times.loc[~stop_times.window.isnull()]

    # Put each trip in the right hour
    stop_times['hour'] = pd.cut(
        departure_time, bins=hours, right=False, labels=hours_labels)

    for c in ['window', 'hour']:
        stop_times[c] = stop_times[c].astype(str)

    return stop_times


def aggregate_by(
        stop_times, labels, index_='stop_id', col='window',
        cutoffs=[0, 6, 9, 15, 19, 22, 24]):
    # Some gtfs feeds only contain direction_id 0, use that as default
    trips_agg = stop_times.pivot_table(
        'trip_id', index=[index_, 'direction_id', col],
        aggfunc='count').reset_index()

    # direction_id is optional, as it is not needed to determine trip frequencies
    # However, if direction_id is NaN, pivot_table will return an empty DataFrame.
    # Therefore, use a sensible default if direction id is not known.
    # Some gtfs feeds only contain direction_id 0, use that as default
    trips_agg.rename(columns={'trip_id': 'ntrips'}, inplace=True)
    start_time = trips_agg.window.apply(lambda x: cutoffs[labels.index(x)])
    end_time = trips_agg.window.apply(lambda x: cutoffs[labels.index(x) + 1])

    trips_agg['frequency'] = ((end_time - start_time)*60 / trips_agg.ntrips)\
        .astype(int)


def add_route_name(line_frequencies, routes):
    # Add the route name
    routes['route_name'] = ''
    if routes.route_short_name.isnull().unique()[0]:
        routes['route_name'] = routes.route_long_name
    elif routes.route_long_name.isnull().unique()[0]:
        routes['route_name'] = routes.route_short_name
    else:
        routes['route_name'] =\
            routes.route_short_name + ' ' + routes.route_long_name

    line_frequencies = pd.merge(
        line_frequencies, routes[['route_id', 'route_name']])

    return line_frequencies


def stops_freq(
        stop_times, stops,
        cutoffs=[0, 6, 9, 15, 19, 22, 24],
        geom=True):

    if 'window' not in stop_times.columns:
        stop_times = window_creation(stop_times, cutoffs)

    labels = label_creation(cutoffs)
    stop_frequencies = aggregate_by(
        stop_times, labels, index_='stop_id',
        col='window', cutoffs=cutoffs)

    if geom is True:
        stops_cols = ['stop_id', 'stop_name', 'geometry']
    else:
        stops_cols = ['stop_id', 'stop_name']
    stop_frequencies = pd.merge(
        stop_frequencies,
        stops[stops_cols], how='left')

    if geom is True:
        stop_frequencies = gpd.GeoDataFrame(
            data=stop_frequencies.drop('geometry', axis=1),
            geometry=stop_frequencies.geometry)

    return stop_frequencies


def lines_freq(
        stop_times, trips, shapes, routes,
        cutoffs=[0, 6, 9, 15, 19, 22, 24],
        geom=True):

    stop_times_first = stop_times.loc[stop_times.stop_sequence == 1, :]

    if 'window' not in stop_times.columns:
        stop_times_first = window_creation(stop_times_first, cutoffs)

    labels = label_creation(cutoffs)
    line_frequencies = aggregate_by(
        stop_times, labels, index_=['route_id', 'shape_id'],
        col='window', cutoffs=cutoffs)

    # Add route name
    line_frequencies = add_route_name(line_frequencies, routes)

    # Do we want a geodataframe?
    if geom is True:
        line_frequencies = pd.merge(line_frequencies, shapes, how='left')
        line_frequencies = gpd.GeoDataFrame(
            data=line_frequencies.drop('geometry', axis=1),
            geometry=line_frequencies.geometry)

    # Clean the df
    keep_these = [
        'route_id', 'route_name', 'direction_id',
        'window', 'frequency', 'ntrips', 'geometry']

    line_frequencies = line_frequencies.loc[
        ~line_frequencies.geometry.isnull(), keep_these]

    return line_frequencies


def segments_freq(
        segments_gdf, stop_times, routes,
        cutoffs=[0, 6, 9, 15, 19, 22, 24],
        geom=True):

    if 'window' not in stop_times.columns:
        stop_times = window_creation(stop_times, cutoffs)

    # Get labels
    labels = label_creation(cutoffs)

    # Aggregate trips
    line_frequencies = aggregate_by(
        stop_times, labels, index_=['route_id', 'stop_id'],
        col='window', cutoffs=cutoffs)

    keep_these = [
        'route_id', 'segment_id', 'start_stop_id', 'start_stop_name',
        'end_stop_name', 'direction_id', 'geometry']

    line_frequencies = pd.merge(
        line_frequencies,
        segments_gdf[keep_these],
        left_on=['route_id', 'stop_id', 'direction_id'],
        right_on=['route_id', 'start_stop_id', 'direction_id'],
        how='left')

    # Remove duplicates after merging
    drop_these = [
        'route_id', 'stop_id', 'direction_id', 'window', 'ntrips', 'frequency',
        'max_trips', 'max_frequency', 'segment_id', 'start_stop_id',
        'start_stop_name', 'end_stop_name']

    line_frequencies.drop_duplicates(subset=drop_these, inplace=True)

    # Add route name
    line_frequencies = add_route_name(line_frequencies, routes)

    # Aggregate for all lines
    data_complete = add_all_lines(line_frequencies, segments_gdf)

    # Do we want a geodataframe?
    if geom is True:
        data_complete = gpd.GeoDataFrame(
            data=data_complete.drop('geometry', axis=1),
            geometry=data_complete.geometry)

    # Clean data
    keep_these = [
        'route_id', 'route_name', 'direction_id', 'segment_id', 'window',
        'frequency', 'ntrips', 'start_stop_id', 'end_stop_id',
        'start_stop_name', 'end_stop_name', 'geometry']

    data_complete = data_complete[keep_these]

    data_complete = data_complete.loc[~data_complete.geometry.isnull()]

    return data_complete
