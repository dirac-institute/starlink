import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

from astropy import coordinates as coord
from astropy.time import Time
import astropy.units as u
import astropy.constants as c
from astropy.modeling import models, fitting
from astropy.visualization import ZScaleInterval, SqrtStretch, ImageNormalize

import lsst.daf.persistence as dafPersist
import lsst.geom


def loadData(repo, dataId):
    """Load one image using the Butler and return relevant info.

    Parameters
    ----------
    repo : `str`
        Butler repository, probably a file path
    dataId : `dict`-like
        Butler Data ID to retrieve exposure
        e.g., {'visit': 12345, 'ccd': 42}

    Returns
    -------
    njyImage : `lsst.afw.Image`
        Image with pixel values of instrumental flux in nJy.
    photoCalib : `lsst.afw.image.PhotoCalib`
    psfRadius : `float`
        Size of PSF, conceptually equivalent to sigma if it were a Gaussian.
        Multiply by 2.355 to get the PSF size as a FWHM instead.
    pixelScale : `lsst.geom.Angle`
        Size of a pixel on the sky, by default in radians.
    visitInfo : `lsst.afw.image.VisitInfo`
        Dict-like metadata for the retrieved image.
    background : `lsst.afw.math.BackgroundList`
        Information about the image background.
    wcs : `lsst.afw.geom.SkyWcs`
        The WCS for the image.
    """
    butler = dafPersist.Butler(repo)
    calexp = butler.get('calexp', dataId=dataId)
    psfRadius = calexp.getPsf().computeShape().getDeterminantRadius()
    badMask = (calexp.mask.array & calexp.mask.getPlaneBitMask("BAD")) == 1
    satMask = (calexp.mask.array & calexp.mask.getPlaneBitMask("SAT")) == 1
    calexp.image.array[badMask] = 0
    calexp.image.array[satMask] = 0
    photoCalib = calexp.getPhotoCalib()
    njyImage = photoCalib.calibrateImage(calexp.maskedImage)
    pixelScale = calexp.getWcs().getPixelScale()
    visitInfo = calexp.getInfo().getVisitInfo()
    background = butler.get('calexpBackground', dataId=dataId)
    wcs = calexp.getWcs()
    return njyImage, photoCalib, psfRadius, pixelScale, visitInfo, background, wcs


def fit_columns(sliced, center=20, doPlot=False):
    """Return the fitted gaussian widths (stddevs)
    of all columns in a numpy array.

    Called by starlinkAnalyze on the `sliced` array to
    measure the satellite trail width at each point along it.

    Parameters
    ---------
    sliced : `np.array` of mini-rotated-image containing
        the satellite trail (a 2D array)
    center : `int` of approximate satellite trail width
    doPlot : `boolean`, default False

    Returns
    -------
    means : `np.array` of Gaussian fit means
    widths : `np.array` of Gaussian fit stddevs

    """
    t_init = models.Gaussian1D(amplitude=2000, mean=center)
    fit_t = fitting.LevMarLSQFitter()
    widths = np.zeros(len(sliced), dtype=np.float32)
    means = np.zeros(len(sliced), dtype=np.float32)
    if doPlot:
        plt.figure()
    for i in range(len(sliced)):
        y = sliced[..., i]
        x = np.arange(0, len(y), dtype=np.float32)
        t = fit_t(t_init, x, y)
        widths[i] = t.stddev.value
        means[i] = t.mean.value
        if doPlot:
            plt.plot(x, y)
            plt.plot(x, t(x))
    return means, widths


def getTrailLength(trailPoint1, trailPoint2):
    """Measure the length of a satellite trail
    between two points in an image.

    Parameters
    ----------
    trailPoint1 : `list` with 2 values
        [x1, y1] coordinates of first point on trail, in pixels
    trailPoint2 : `list` with 2 values
        [x2, y2] coordinates of second point on trail, in pixels

    Returns
    -------
    trailLength : `int`
        Length of trail, in pixels
    """
    trailLength = int(np.sqrt((trailPoint2[1] - trailPoint1[1])**2 +
                              (trailPoint2[0] - trailPoint1[0])**2))
    return trailLength


def makeTrailHorizontal(imageArray, trailPoint1, trailPoint2, trailWidth):
    """Rotate an image containing a satellite trail so it is horizontal.

    Parameters
    ----------
    imageArray : np.array
        2D array containing image data.
    trailPoint1 : `list` with 2 values
        [x1, y1] coordinates of first point on trail, in pixels
    trailPoint2 : `list` with 2 values
        [x2, y2] coordinates of second point on trail, in pixels
    trailWidth : `int`
        Width of trail, in pixels

    Returns
    -------
    rotatedArray : np.array
        2D array containing the same data as imageArray, but rotated
        so that the satellite trail is horizontal.
    trailRotX : `int`
        x-coordinate of trailPoint1 in the rotated frame
    trailRotY : `int`
        y-coordinate of trailPoint1 in the rotated frame
    sliced : np.array
        2D array, sized (trailLength x 2*trailWidth), containing image data
        of a small region which includes the satellite trail
    """
    angle = np.arctan2(trailPoint2[1] - trailPoint1[1],
                       trailPoint2[0] - trailPoint1[0]) * u.rad
    slope = (trailPoint2[1] - trailPoint1[1])/(trailPoint2[0] - trailPoint1[0])
    yint = trailPoint1[1] - slope*trailPoint1[0]
    xmax = len(imageArray[0])
    ymax = len(imageArray)
    rotatedArray = rotate(imageArray, angle)
    trailRotYPart1 = np.abs(xmax * np.sin(angle))
    trailRotYPart2 = (xmax*slope + yint) * np.cos(angle)
    trailRotY = int(trailRotYPart1 + trailRotYPart2)
    trailRotXPart1 = np.abs((ymax - yint) * np.sin(angle))
    trailRotXPart2 = np.abs(trailPoint1[0]/np.cos(angle))
    trailRotX = int(trailRotXPart1 + trailRotXPart2)
    trailLength = getTrailLength(trailPoint1, trailPoint2)
    sliced = rotatedArray[trailRotY - trailWidth:trailRotY + trailWidth,
                          trailRotX:trailRotX + trailLength]
    return rotatedArray, trailRotX, trailRotY, sliced


def plotSatelliteTrail(imageArray, trailPoint1, trailPoint2, trailWidth):
    """Make three figures to illustrate the original and rotated image,
    summed flux profile of the satellite trail, and flux along the trail.

    Parameters
    ----------
    imageArray : np.array
        2D array containing image data.
    trailPoint1 : `list` with 2 values
        [x1, y1] coordinates of first point on trail, in pixels
    trailPoint2 : `list` with 2 values
        [x2, y2] coordinates of second point on trail, in pixels
    trailWidth : `int`
        Width of trail, in pixels
    """
    rotatedInfo = makeTrailHorizontal(imageArray, trailPoint1, trailPoint2, trailWidth)
    rotatedArray = rotatedInfo[0]
    trailRotX = rotatedInfo[1]
    trailRotY = rotatedInfo[2]
    sliced = rotatedInfo[3]
    trailLength = getTrailLength(trailPoint1, trailPoint2)
    norm = ImageNormalize(imageArray, interval=ZScaleInterval(), stretch=SqrtStretch())
    fig1 = plt.figure(figsize=(8, 4))
    fig1.add_subplot(121)
    plt.imshow(imageArray, cmap='gray', norm=norm, origin='lower')
    plt.plot([trailPoint1[0], trailPoint2[0]], [trailPoint1[1], trailPoint2[1]],
             ls=':', color='C0', lw=2)
    plt.title('Original image with satellite trail')
    fig1.add_subplot(122)
    plt.imshow(rotatedArray, cmap='gray', norm=norm, origin='lower')
    plt.axhline(y=trailRotY - trailWidth, ls=':', color='C1', lw=2)
    plt.axhline(y=trailRotY + trailWidth, ls=':', color='C1', lw=2)
    plt.axhline(y=trailRotY, ls=':', color='C0', lw=2)
    plt.plot(trailRotX, trailRotY, marker='o', color='C4')
    plt.plot(trailRotX + trailLength, trailRotY, marker='o', color='C4')
    plt.title('Rotated image with horizontal satellite trail')

    fig2 = plt.figure(figsize=(8, 4))
    ax2 = fig2.subplots()
    ax2.plot(sliced.sum(axis=1), marker='o')
    plt.xlabel('Pixel index')
    plt.ylabel('Flux (nJy)')
    plt.title('Summed flux profile')

    fig3 = plt.figure(figsize=(8, 4))
    ax3 = fig3.subplots()
    ax3.plot(sliced.sum(axis=0))
    plt.xlabel('Rotated X pixel position')
    plt.ylabel('Flux (nJy)')
    plt.title('Flux along the trail')


def computeAngularSpeed(visitInfo, wcs, trailPoint1, trailPoint2, height=550*u.km):
    """
    Compute the angular speed of a satellite.

    Parameters
    ------
    visitInfo : `lsst.afw.image.VisitInfo`
        Dict-like metadata for the image.
    wcs : `lsst.afw.geom.SkyWcs`
    The WCS for the image.
    trailPoint1 : `list` with 2 values
        [x1, y1] coordinates of first point on trail, in pixels
    trailPoint2 : `list` with 2 values
        [x2, y2] coordinates of second point on trail, in pixels
    height : `astropy.Quantity`, optional
        Altitude or height of satellite above the surface of Earth
        Default is 550 km

    Returns
    -------
    speed : `astropy.Quantity`
        Angular speed as seen from the surface of Earth (angle per time)
    """
    airmass = visitInfo.getBoresightAirmass()
    omega = np.sqrt(c.G * c.M_earth/(c.R_earth + height)**3)  # angular speed from center of earth
    orbitSpeed = omega * (c.R_earth + height)
    x, d = computeDistanceToSatellite(airmass, height)
    # x is the angle between line of sight and (Radius of the Earth + height)
    # d is the distance between the satellite and an observer on earth
    # Account for the fact that the satellite trail does not pass through zenith
    az1, alt1 = trailPointToAzAlt(visitInfo, wcs, trailPoint1)
    az2, alt2 = trailPointToAzAlt(visitInfo, wcs, trailPoint2)
    if (alt1 == 0 and alt2 == 0 and az1 == 0 and az2 == 0):
        tanSpeed = orbitSpeed * np.cos(x)  # project orbitSpeed to perpendicular to line of sight
        # angleHorizon = 90
    else:
        tanTheta = (az2 - az1)/(alt2 - alt1)
        xSquare = 1./(1 + np.tan(x)**2 + tanTheta**2)
        zSquare = xSquare*np.tan(x)**2
        tanSpeed = orbitSpeed * np.sqrt(1 - zSquare)
        # angleHorizon = 90 - abs(np.degrees(np.arctan(tanTheta)))
    angularSpeed = tanSpeed/d * u.rad  # angular speed from surface of Earth

    return angularSpeed


def computeDistanceToSatellite(airmass, height):
    """Given airmass and range, compute how far away a satellite actually is.

    Parameters
    ----------
    airmass : `float`
        Airmass of the observation.
    height : `astropy.Quantity`
        Altitude or height of satellite above the surface of Earth

    Returns
    -------
    x : `float`
        The (small) angle between line of sight and (Radius of the Earth + height)
    d: `astropy.Quantity`
        The distance between the satellite and an observer on earth
    """
    zangle = np.arccos(1./airmass) * u.rad
    x = np.arcsin(c.R_earth * np.sin(zangle)/(c.R_earth + height))
    if np.isclose(x, 0):
        d = height
    else:
        d = np.sin(zangle - x) * c.R_earth/np.sin(x)
    return x, d


def computeSatelliteSize(avg_fwhm, psfRadius, airmass, pixelScale,
                         D_mirror=4*u.m, height=550*u.km):
    """Compute the physical size of a satellite at zenith.

    Parameters
    ----------
    avg_fwhm : `float`
        Average FWHM of satellite trail width on an image, in pixels.
    psfRadius : `float`
        Size of PSF, conceptually equivalent to sigma if it were a Gaussian.
        Multiply by 2.355 to get the PSF size as a FWHM instead.
    airmass : `float`
        Airmass of the observation.
    pixelScale : `lsst.geom.Angle`
        Size of a pixel on the sky, by default in radians.
    D_mirror
    height : `astropy.Quantity`, optional
        Altitude or height of satellite above the surface of Earth
        Default is 550 km

    Returns
    -------
    sat_size : `astropy.Quantity`
        The derived physical size of the satellite at zenith.
    """
    avg_fwhm = avg_fwhm * pixelScale.asRadians() * u.rad
    psfRadius = psfRadius * pixelScale.asRadians() * u.rad
    x, d = computeDistanceToSatellite(airmass, height)
    angular_size_squared = avg_fwhm**2 - (psfRadius*2.355)**2  # in square radians
    # note that angular_size^2 = (D_sat^2 + D_mirror^2) / d^2
    sat_size_squared = (angular_size_squared.value * d**2) - D_mirror**2
    sat_size = np.sqrt(sat_size_squared)
    # note we assume tan(angular_size) ~ angular_size as D << d
    return d, sat_size


def trailPointToAzAlt(visitInfo, wcs, trailPoint, loc="Cerro Tololo Interamerican Observatory"):
    """Get the Azimuth and Altitude corresponding to an x, y image position.

    Parameters
    ----------
    visitInfo : `lsst.afw.image.VisitInfo`
        Dict-like metadata for the image.
    wcs : `lsst.afw.geom.SkyWcs`
        The WCS for the image.
    trailPoint : `list` with 2 values
            [x, y] coordinates of a point in image, in pixels
    loc : `str`, optional
        Default is "Cerro Tololo Interamerican Observatory"

    """
    dateObs = visitInfo.getDate().toPython()
    dateObsAstropy = Time(dateObs)
    sky_coord_afw = wcs.pixelToSky(lsst.geom.Point2D(trailPoint[0], trailPoint[1]))
    sky_coord = coord.SkyCoord(sky_coord_afw.getRa().asDegrees()*u.deg,
                               sky_coord_afw.getDec().asDegrees()*u.deg)
    location = coord.EarthLocation.of_site(loc)
    aa_frame = coord.AltAz(obstime=dateObsAstropy, location=location)
    aa = sky_coord.transform_to(aa_frame)

    return aa.az, aa.alt


def starlinkAnalyze(repo, dataId, trailPoint1, trailPoint2, trailWidth=20):
    """Analyze an image processed with the LSST Science Pipelines
    containing a satellite trail.

    Parameters
    ----------
    repo : `str`
        Butler repository, probably a file path
    dataId : `dict`-like
        Butler Data ID to retrieve exposure
        e.g., {'visit': 12345, 'ccd': 42}
    trailPoint1 : `list` with 2 values
        [x1, y1] coordinates of first point on trail, in pixels
    trailPoint2 : `list` with 2 values
        [x2, y2] coordinates of second point on trail, in pixels
    trailWidth : `int`
        Approximate half-width of satellite trail, in pixels, optional
        Default is 20

    Returns
    -------
    results : `dict`
        Results dict containing over a dozen relevant values
        Note that all numeric quantities are rounded to 2 decimal places

    Notes
    -----
    For best results, display or plot the image beforehand,
    and manually choose trailPoint1 and trailPoint2 so they are
    at opposite ends of the satellite trail.

    """
    # Load data
    njyImage, photoCalib, psfRadius, pixelScale, visitInfo, background, wcs = loadData(repo, dataId)

    # Get sun location and phase angle
    boresight_raDec = visitInfo.getBoresightRaDec()
    dateObs = visitInfo.getDate().toPython()
    dateObsAstropy = Time(dateObs)
    sky_coord = coord.SkyCoord(boresight_raDec.getRa().asDegrees()*u.deg,
                               boresight_raDec.getDec().asDegrees()*u.deg)
    location = coord.EarthLocation.of_site('Cerro Tololo Interamerican Observatory')
    aa_frame = coord.AltAz(obstime=dateObsAstropy, location=location)
    sun_coord = coord.get_sun(dateObsAstropy)
    sun_aa = sun_coord.transform_to(aa_frame)
    phase_angle = sky_coord.separation(sun_coord)

    # Get background info since it was subtracted from image
    background_in_nJy = photoCalib.instFluxToNanojansky(background.getImage().getArray().mean())
    background_per_area = (background_in_nJy) / pixelScale.asArcseconds()**2
    background_mag_per_arcsec_squared = (background_per_area*u.nJy).to(u.ABmag).value

    # Rotate image data so the trail is horizontal
    rotatedInfo = makeTrailHorizontal(njyImage.image.array, trailPoint1, trailPoint2, trailWidth)
    sliced = rotatedInfo[3]
    trailLength = getTrailLength(trailPoint1, trailPoint2)

    # Make sure nothing horrible is happening with outliers in the trail
    FcorrFactor = np.median(sliced.sum(axis=0))/np.mean(sliced.sum(axis=0))
    assert np.abs(1 - FcorrFactor) < 0.1

    # Make plots of satellite trail
    plotSatelliteTrail(njyImage.image.array, trailPoint1, trailPoint2, trailWidth)

    # Calculate trail brightness
    summed_flux = sliced.sum()
    means, widths = fit_columns(sliced, center=trailWidth)
    avg_fwhm = 2.355 * np.mean(widths)
    flux_per_pixel = summed_flux / (avg_fwhm*trailLength)  # nJy per square pixel
    flux_per_arcsec_squared = flux_per_pixel / pixelScale.asArcseconds()**2  # nJy per arcsec
    mag_per_arcsec_squared = (flux_per_arcsec_squared*u.nJy).to(u.ABmag).value

    # Some correction factors
    # correction for exposure time
    corr = 2.5 * np.log10(visitInfo.getExposureTime())
    # distance traveled in sky by satellite in 1 second
    airmass = visitInfo.getBoresightAirmass()
    angularSpeed = computeAngularSpeed(visitInfo, wcs, trailPoint1, trailPoint2)
    dist_1_sec = angularSpeed * 1.*u.s
    # correction for sky covered in 1 second
    corr2 = 2.5 * np.log10(dist_1_sec.to(u.arcsec).value * avg_fwhm * pixelScale.asArcseconds())
    stationary_mag = mag_per_arcsec_squared - corr - corr2

    # Effective satellite size
    dist_to_sat, sat_size = computeSatelliteSize(avg_fwhm, psfRadius, airmass, pixelScale)

    # Save relevant information to results dict
    results = dict()
    results['Visit'] = dataId['visit']
    results['CCD'] = dataId['ccd']
    results['Date and time (UTC)'] = dateObsAstropy
    results['Exposure time'] = visitInfo.getExposureTime() * u.s
    results['Boresight Az'] = visitInfo.getBoresightAzAlt()[0].asDegrees() * u.deg
    results['Boresight Alt'] = visitInfo.getBoresightAzAlt()[1].asDegrees() * u.deg
    results['Sun Az'] = sun_aa.az.to(u.deg).value * u.deg
    results['Sun Alt'] = sun_aa.alt.to(u.deg).value * u.deg
    results['Phase angle'] = phase_angle.to(u.deg).value * u.deg
    results['Airmass'] = airmass
    results['Image PSF radius'] = psfRadius * pixelScale.asArcseconds() * u.arcsec
    results['Image PSF FWHM'] = psfRadius*2.355 * pixelScale.asArcseconds() * u.arcsec
    results['Background (mag)'] = background_mag_per_arcsec_squared * u.arcsec**(-2)
    results['Summed trail flux'] = summed_flux * u.nJy
    results['Trail FWHM'] = avg_fwhm * pixelScale.asArcseconds() * u.arcsec
    results['Trail length'] = trailLength * pixelScale.asArcseconds() * u.arcsec
    results['Raw trail flux'] = flux_per_arcsec_squared * u.nJy * u.arcsec**(-2)
    results['Raw trail (mag)'] = mag_per_arcsec_squared * u.arcsec**(-2)
    results['Corrected trail (mag)'] = (mag_per_arcsec_squared - corr) * u.arcsec**(-2)
    results['Angular speed'] = angularSpeed.to(u.deg / u.s)
    results['Stationary magnitude'] = stationary_mag
    results['Corrected stationary mag'] = (stationary_mag - 5*np.log10(airmass))
    results['Derived satellite size'] = sat_size.to(u.m)
    results['Distance to satellite'] = dist_to_sat.to(u.km)

    # Round overly precise values
    for key, value in results.items():
        if isinstance(value, u.Quantity):
            results[key] = value.round(2)
        elif isinstance(value, float):
            results[key] = np.round(value, 2)

    return results
