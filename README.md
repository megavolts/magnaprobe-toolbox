# magnaprobe-toolsbox

Magnaprobe toolbox provides basic function to process and analyse MangaProbe snowdepth data.

The MagnaProbe is an automatic device design to measure both depth and position simultaneously. When use correctly, each button click record a geolocated snow depth. [Sturm & Holmgren (2018)](https://doi.org/10.1029/2018WR023559) for instrument details.

## Importation
Raw MagnaProbe data is saved in a comma separated value table of the form ('.dat'):

| TOA5 | 48066 | CR800 | 48066 | CR800.Std.32.05 | CPU:Probe48066-20140419.CR8 | 14760 | OperatorView |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|------|-------|-------|-------|-----------------|-----------------------------|-------|--------------|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|
| TIMESTAMP | RECORD | Counter | DepthCm | BattVolts | latitude_a | latitude_b | Longitude_a | Longitude_b | fix_quality | nmbr_satellites | HDOP | altitudeB | DepthVolts | LatitudeDDDDD | LongitudeDDDDD | month | dayofmonth | hourofday | minutes | seconds | microseconds |
| TS | RN |  |  |  | degrees | minutes | degrees | minutes | unitless |  |  |  |  |  |  |  |  |  |  |  |  |
| 2024-06-07 13:22:57.28 | 14267 | 311000 | 119.3 | 13.71 | 71 | 20.9663 | -156 | -31.4397 | 2 | 11 | 0.7 | -1.2 | 7.601 | 0.3494383 | -0.523995 | 6 | 7 | 13 | 22 | 56 | 600000 |
| 2024-06-07 13:22:58.38 | 14268 | 311001 | 119.6 | 13.71 | 71 | 20.9663 | -156 | -31.4396 | 2 | 11 | 0.7 | -1.1 | 7.615 | 0.3494383 | -0.5239933 | 6 | 7 | 13 | 22 | 57 | 700000 |
| 2024-06-07 13:23:06 | 14269 | 311002 | 5.334 | 13.71 | 71 | 20.9662 | -156 | -31.4401 | 2 | 11 | 0.7 | -1.1 | 0.346 | 0.3494367 | -0.5240016 | 6 | 7 | 13 | 23 | 5 | 310000 |

Due to the CR800 Campbell datalogger memory limitation, latitude and longitude cannot be saved directly as a decimal coordinate. Each coordinate is divided in two fields. The first one (_a) contains the full degree integer, and the second one (_b) contains the decimal minute. Thus we can compute Latitude and Longitude as:

$$ Latitude = Latitude_a + Latitude_b/60 $$

and 

$$ Longitude = Longitude_a + Longitude_b/60 $$

During the importation snow depth stored orignally in centimeter are converted to meter. 


## Geolocation
Additionnal functions are provided to
 - project coordinate into local UTM system, define by their EPSG number. By default WGS84 is used (EPSG 4326)
 - compute distance between each consecutive points, and relatively to the transect start point

## Analysis
Some basic analysis tools are provided to
- compute basic statistic
- compute semivariogram

### Dependencies
* pandas
* pyproj
* matplotlib
* numpy






[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
