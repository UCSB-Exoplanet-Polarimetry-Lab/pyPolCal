import numpy as np
from photutils.aperture import RectangularAperture
wavelength_bins = np.array([1159.5614, 1199.6971, 1241.2219, 1284.184 , 1328.6331, 1374.6208,
1422.2002, 1471.4264, 1522.3565, 1575.0495, 1629.5663, 1685.9701,
1744.3261, 1804.7021, 1867.1678, 1931.7956, 1998.6603, 2067.8395,
2139.4131, 2213.4641, 2290.0781, 2369.3441])

centroid_lbeam = [71.75, 86.25]
centroid_rbeam = [131.5, 116.25]
aperture_width = 44.47634202584561
aperture_height = 112.3750880855165
theta = 0.46326596610192305

charis_aperture_l = RectangularAperture(centroid_lbeam, aperture_width, aperture_height, theta=theta)
charis_aperture_r = RectangularAperture(centroid_rbeam, aperture_width, aperture_height, theta=theta)