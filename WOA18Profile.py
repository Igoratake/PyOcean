# -*- coding: iso-8859-15 -*-
# -*- coding: utf-8 -*-


from xarray import open_mfdataset, open_dataset, merge
from gsw import rho_t_exact, p_from_z
import cmocean.cm as cmo
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from numpy import linspace, array, ones, any
from PIL import Image
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from TTutils.Utils import *
from numpy import asarray
from TTutils.logo import *
from TTutils.data import *
from .data import *


def WOA18Profile_TSD(Time, LonPoint, LatPoint,
                     SeasonClimatology=True, Depths=None):
    '''
    Gets WOA18 TSD climatology profile from a single point

    Time: int,list
        Value or list months or season value. If pass a list it returns
        a mean value. For season values SeasonClimatology must be True. And the
        season values follow bellow.
            1 = summer = mean(21Dec - 21Mar)
            2 = autum  = mean(21Mar - 21Jun)
            3 = winter = mean(21Jun - 21Sep)
            4 = spring = mean(21Sep- 21Dec)
    LonPoint/LatPoint: float or int default None
        longitude/latitude coordinate point. If will not value given the file
        must have one single point.
    SeasonClimatology: bool, default True
        Define if Time number is represents a month (False) or a season (True)
    Depths: str,int,float or list, default None
        Depth value (or list of) to selects from depth data. The value must be
        exacly the same in NetCDF file. For know depth values in specific point
        pass 'values' as string.

    output:
        List with Dataframe with Salinity, Temperature, Density and depth
        axis and Longitude and Latitude in a tuple.The value could be the mean
        or absolut values dependsif Time input is a list or a single value.

    Example:

    >>> Time=1
    >>> LonPoint=-40
    >>> LatPoint=-30
    >>> df = WOA13Profile_TSD(Time,LonPoint,LatPoint)
    >>> df.head()
                    Salinity  Temperature
        depth
        0.0    36.188713    20.768999
        5.0    36.191601    20.742691
        10.0   36.199913    20.724991
        15.0   36.202412    20.692410
        20.0   36.203888    20.637800

    >>> depth = WOA13Profile_TSD(Time,LonPoint,LatPoint,Depths='values')
    >>> depth
    Float64Index([   0.0,    5.0,   10.0,   15.0,   20.0,   25.0,   30.0, 35.0,
                40.0,   45.0,   50.0,   55.0,   60.0,   65.0,   70.0,   75.0,
                80.0,   85.0,   90.0,   95.0,  100.0,  125.0,  150.0,  175.0,
               200.0,  225.0,  250.0,  275.0,  300.0,  325.0,  350.0,  375.0,
               400.0,  425.0,  450.0,  475.0,  500.0,  550.0,  600.0,  650.0,
               700.0,  750.0,  800.0,  850.0,  900.0,  950.0, 1000.0, 1050.0,
              1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1350.0, 1400.0, 1450.0,
              1500.0, 1550.0, 1600.0, 1650.0, 1700.0, 1750.0, 1800.0, 1850.0,
              1900.0, 1950.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0,
              2600.0, 2700.0, 2800.0, 2900.0, 3000.0, 3100.0, 3200.0, 3300.0],
             dtype='float64', name=u'depth')
    '''

    URLPrefix = 'https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa/'
    URLSufix = '/decav/0.25/woa18_decav_'
    extention = '_04.nc'

    TempPrefix = URLPrefix + 'temperature' + URLSufix + 't'
    SalPrefix = URLPrefix + 'salinity' + URLSufix + 's'

    if not isinstance(Time, list):
        Time = [Time]

    if SeasonClimatology:
        Time = list(map(lambda x: x + 12, Time))

    Tfnames = [TempPrefix + '{:02d}'.format(i) + extention for i in Time]
    Sfnames = [SalPrefix + '{:02d}'.format(i) + extention for i in Time]

    fnames = Tfnames + Sfnames

    del Tfnames, Sfnames

    ds = open_mfdataset(fnames, decode_times=False)
    ds = ds[['t_an', 's_an']].sel(lon=LonPoint, lat=LatPoint, method='nearest')
    lat = ds.lat.values
    lon = ds.lon.values
    ds = ds.squeeze()

    if len(Time) > 1:
        ds = ds.mean('time').squeeze()

    rename = {'s_an': 'Salinity', 't_an': 'Temperature'}
    df = ds.to_dataframe().dropna().rename(columns=rename)

    if isinstance(Depths, (list, int, float)):
        df = df.loc[Depths, :]
    Pressure = p_from_z(-df.index.values, lat)
    density = rho_t_exact(df.Salinity.values, df.Temperature.values, Pressure)
    df.loc[:, 'Density'] = density
    if 'time' in df.columns:
        df.drop('time', axis=1, inplace=True)

    ds.close()

    del ds

    return [df, (lon, lat)]


def Woa18TSDMapPlot(Time, OutName='field',
                    LonRange=[-50, -39],
                    LatRange=[-30, -22],
                    Depths=None,
                    projection=None,
                    ShpData=None,
                    limits=None,
                    SeasonClimatology=True,
                    LogoLocatoin=[.15, .6, .15, .15],
                    TempRange=[15, 30],
                    SalRange=[32, 37.5],
                    RhoRange=[1021, 1027.5],
                    **kwargs):

    # definition variables
    URLPrefix = 'https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa/WOA18'
    URLSufix = '/decav/0.25/woa18_decav_'
    extention = '_04.nc'
    basemap = BasemapDataBrasil()

    TempPrefix = URLPrefix + 'temperature' + URLSufix + 't'
    SalPrefix = URLPrefix + 'salinity' + URLSufix + 's'

    units = [r'$^{o}$C', 'psu', r'kg/m$^{3}$']
    # axis formats
    lon_formatter = LongitudeFormatter(number_format='.1f',
                                       degree_symbol='',
                                       dateline_direction_label=True)
    lat_formatter = LatitudeFormatter(number_format='.1f',
                                      degree_symbol='')

    # checks user inputs
    if not projection:
        projection = ccrs.Miller()

    LonMin = min(LonRange)
    LonMax = max(LonRange)
    LatMin = min(LatRange)
    LatMax = max(LatRange)

    if not Depths:
        Depths = [0]
        ByDim = None
    elif isinstance(Depths, (float, int)):
        Depths = [Depths]
        ByDim = None
    elif isinstance(Depths, list) and len(Dephts) == 1:
        ByDim = None
    else:
        ByDim = 'depth'

    if not isinstance(Time, list):
        Time = [Time]

    if SeasonClimatology:
        Time = list(map(lambda x: x + 12, Time))

    # processig
    Tfnames = [TempPrefix + '{:02d}'.format(i) + extention for i in Time]
    Sfnames = [SalPrefix + '{:02d}'.format(i) + extention for i in Time]

    fnames = Tfnames + Sfnames

    del Tfnames, Sfnames

    # acess opendap
    ds = open_mfdataset(fnames, decode_times=False)
    # cuts data
    gebco = open_dataset(GebcoData())
    gebco = gebco.ELEVATION.sel(LON10801_19200=slice(LonMin, LonMax),
                                LAT3601_13200=slice(LatMin, LatMax))

    ds.rename({'t_an': 'Temperatura', 's_an': 'Salinidade'}, inplace=True)
    ds = ds[['Temperatura', 'Salinidade']].sel(
        lon=slice(LonMin, LonMax),
        lat=slice(LatMin, LatMax),
        depth=Depths)
    ds = ds.squeeze()

    # gets coordinates from data and prepare
    lat = ds.lat.values
    lon = ds.lon.values
    LatMax = ds.lat.values.max() - 0.25
    LatMin = ds.lat.values.min() + 0.25
    LonMax = ds.lon.values.max() - 0.25
    LonMin = ds.lon.values.min() + 0.25
    # prepare plot thicks
    lon = linspace(LonMin, LonMax, 5)
    lat = linspace(LatMin, LatMax, 5)

    # if necessary calculates mean in time
    if len(Time) > 1:
        ds = ds.mean('time').squeeze()

    # data interpolate

    ds = FillDataSetNaNs(ds, ['Temperatura', 'Salinidade'], ByDim)
    density = ds['Temperatura'].to_dataset(name='Densidade')
    # density = ones(ds.Temperatura.shape)
    density['Densidade'].values = rho_t_exact(ds.Salinidade, ds.Temperatura, 0)

    ds = merge([ds, density])

    cmaps = [cmo.thermal, cmo.haline, cmo.dense]
    Ranges = [TempRange, SalRange, RhoRange]
    for depth in Depths:
        for prop, cmap, unit, Range in zip(
                ds.data_vars.keys(), cmaps, units, Ranges):

            fname = '_'.join([OutName, prop, str(depth)])
            vmin = Range[0]
            vmax = Range[1]
            ax = plt.axes(projection=projection)
            if len(Depths) > 1:
                ds.sel(depth=depth, method='nearest').squeeze()[prop].plot.contourf(
                    levels=50,
                    ax=ax,
                    transform=projection,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={
                        'label': unit,
                        'format': "%.2f"
                    })

            else:
                ds.squeeze()[prop].plot.contourf(
                    levels=50,
                    ax=ax,
                    transform=projection,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    cbar_kwargs={
                        'label': unit
                    })

            if depth > 50:
                GrayCmap, norm = from_levels_and_colors(
                    [-depth, depth], ['#d8dcd6'])
                gebco.where((gebco >= -depth) & (gebco <= depth)).plot.contourf(
                    norm=norm,
                    cmap=GrayCmap,
                    add_colorbar=False)

            shape_feature = ShapelyFeature(
                Reader(basemap).geometries(),
                ccrs.PlateCarree(),
                edgecolor='#6E6E6E',
                facecolor='#FFFFEB')

            if isinstance(ShpData, str):
                PointLocation = ShapelyFeature(
                    Reader(ShpData).geometries(),
                    ccrs.PlateCarree(),
                    edgecolor='#000000',
                    facecolor='#000000')
                ax.add_feature(PointLocation, zorder=10)

            ax.add_feature(shape_feature)
            ax.gridlines()
            ax.set_xticks(lon, crs=projection)
            ax.set_yticks(lat, crs=projection)
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(' a '.join([prop, str(depth)]) + 'm')
            a = plt.axes(LogoLocatoin, facecolor='None')
            im = plt.imshow(array(Image.open(GetLogo())))
            plt.yticks(rotation='vertical')
            plt.axis('off')
            plt.setp(a, xticks=[], yticks=[])

            plt.savefig(fname + '.png', dpi=300, bbox_inches='tight')
            plt.close()


def Woa18RegionDepth(LonPoint,
                     LatPoint,
                     SeasonClimatology=True,
                     Depth=None):
    '''
    Returns maximum depth with valid data in a region around a point

    LonPoint/LatPoint: float or int
        longitude/latitude coordinate point.
    SeasonClimatology: bool, default True
        Define if Time number is represents a month (False) or a season (True)
    Depths: int or float default None
        Depth value to selects coordinates with minimum data. The value
        must be
        exacly the same in NetCDF file. For know depth values in specific point
        pass 'values' as string.

    output:
        DataFrame with depth values, columns equal longitude and index
        are latitudes.NaN values, is because thre isn't values bigger than that
    '''
    URLPrefix = 'https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa/WOA18'
    URLSufix = '/decav/0.25/woa18_decav_'
    extention = '_04.nc'

    TempPrefix = URLPrefix + 'temperature' + URLSufix + 't'
    if SeasonClimatology:
        Time = '13'
    else:
        Time = '01'

    fname = [TempPrefix + Time + extention]

    ds = open_mfdataset(fname, decode_times=False).squeeze()
    ds = ds.drop('time')

    LonMin = LonPoint - 0.255
    LonMax = LonPoint + 0.255
    LatMin = LatPoint - 0.255
    LatMax = LatPoint + 0.255

    if not Depth:
        Depth = 0

    depths = ds.sel(
        lon=slice(LonMin, LonMax),
        lat=slice(LatMin, LatMax)).squeeze()
    depths = depths['t_an'].to_dataframe().dropna()
    depths = depths.query('depth>={}'.format(Depth)).unstack([1, 2])
    depths = depths.apply(lambda x: x.notnull() * x.index, axis=0)
    depths = depths.max(axis=0).unstack(1)
    depths.index = depths.index.droplevel()

    return depths.T
