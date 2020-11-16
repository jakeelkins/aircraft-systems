wgs84 = wgs84Ellipsoid;

lat0 = (180/pi)*1.369987e-2;
lon0 = (180/pi)*-2.83742958e-2;
h0 = 3.20995e+2;

%lat0 = 0;
%lon0 = 0;
%h0 = 0;

lat = (180/pi)*1.36784621e-2;
lon = (180/pi)*-2.81966732e-2;
h = 2.71732375e+2;

[n, e, d] = geodetic2ned(lat,lon,h,lat0,lon0,h0,wgs84)