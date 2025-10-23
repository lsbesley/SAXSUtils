from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import fabio
import os
import pickle

def open_edf(filename):
    # Open edf data
    return fabio.open(filename)


def get_real_q(data_filename,data_size,custom_SD=55.0004):
    data2dIMG = open_edf(data_filename)
    #first we need to convert the existing 0-1 q to n. pixels
    SD = custom_SD#float(data2dIMG.header.get('det0y'))#sample detector distance
    pixel_size = float(data2dIMG.header.get('PSize_1'))#sample detector distance
    wavelength = float(data2dIMG.header.get('Wavelength'))#sample detector distance
    print("Sample to detector distance:", SD, "mm")
    print("Pixel Size:", pixel_size, "m,",pixel_size*1e6,"um")
    print("Wavelength:", wavelength, "m")
    
    n_pixels = np.arange(data_size)
    pixel_real_sizes = n_pixels*pixel_size
    # pixel_real_sizes_SAXS = np.arange(100)*pixel_size
    two_theta = np.arctan(pixel_real_sizes/(SD*1e-3)) #in radians
    # two_theta_SAXS = np.arctan(pixel_real_sizes_SAXS/(SD_SAXS*1e-3)) #in radians
    real_q = 4*np.pi/(wavelength*1e10) * np.sin(two_theta/2)
    # real_q_SAXS = 4*np.pi/(wavelength*1e10) * np.sin(two_theta_SAXS/2)
    return real_q




def multi_gaussian(x, *params):
    """
    Generalized Gaussian fitting function for an arbitrary number of peaks.

    Parameters:
    - x: Input data (independent variable).
    - params: A flat list of parameters, where each Gaussian is defined by
      (amp, mean, stddev, c). The list should have a length that is a
      multiple of 4.

    Returns:
    - The sum of all Gaussian peaks.
    """
    if len(params) % 4 != 0:
        raise ValueError("The number of parameters must be a multiple of 4 (amp, mean, stddev, c for each Gaussian).")
    
    num_peaks = len(params) // 4
    result = np.zeros_like(x)
    
    for i in range(num_peaks):
        amp, mean, stddev, c = params[4*i:4*(i+1)]
        result += amp * np.exp(-0.5 * ((x - mean) / stddev) ** 2) + c
    
    return result

def multi_gaussian_linbg(x, *params):
    """
    Generalized Gaussian fitting function for an arbitrary number of peaks.

    Parameters:
    - x: Input data (independent variable).
    - params: A flat list of parameters, where each Gaussian is defined by
      (amp, mean, stddev, c). The list should have a length that is a
      multiple of 4.

    Returns:
    - The sum of all Gaussian peaks.
    """
    if len(params) % 4 != 2:
        raise ValueError("The number of parameters must be a multiple of 4 (amp, mean, stddev, c for each Gaussian).")
    
    num_peaks = len(params) // 4
    result = np.zeros_like(x)
    
    for i in range(num_peaks):
        amp, mean, stddev, c = params[4*i:4*(i+1)]
        result += amp * np.exp(-0.5 * ((x - mean) / stddev) ** 2) + c
    m,c = params[-2:]
    result += m*x+c
    return result



def fit_waxs_arbitrary_n_peaks(initial_guess,bounds,xdata,ydata,q_ROI,n_gauss_peaks,sample_label='sample',fig_name="figure",savefig=False,filepath=None):
    """
    Fit a selected q-range of WAXS data with an arbitrary number of Gaussian peaks.

    This function fits multiple Gaussian peaks (without a linear background)
    to a selected region of interest (ROI) in 1D WAXS intensity data.
    It prints the fitted parameters, plots the data and individual peak components,
    and optionally saves the figure.

    Parameters
    ----------
    initial_guess : list or ndarray
        Initial guess for all fitting parameters. Should contain 4 * n_gauss_peaks elements:
        [amp1, mean1, stddev1, c1, amp2, mean2, stddev2, c2, ...].
    bounds : 2-tuple of array-like
        Lower and upper bounds for parameters, as required by `scipy.optimize.curve_fit`.
    xdata : ndarray
        Array of q-values.
    ydata : ndarray
        Array of corresponding intensity values.
    q_ROI : tuple of int
        Indices defining the range of q-values to fit, e.g. (start_idx, end_idx).
    n_gauss_peaks : int
        Number of Gaussian peaks to fit.
    sample_label : str, optional
        Label used as the plot title. Default is 'sample'.
    fig_name : str, optional
        Name of the output figure file if saving is enabled. Default is 'figure'.
    savefig : bool, optional
        Whether to save the resulting plot as an image. Default is False.
    filepath : str or None, optional
        Directory to save the figure. If None, saves in the current working directory.

    Outputs
    -------
    - Prints a formatted table of fitted Gaussian parameters (amplitude, mean, stddev, c).
    - Displays a plot with data points, the combined fit, and individual Gaussian peaks.

    Notes
    -----
    - Uses the function `multi_gaussian(x, *params)` to define the model.
    - Each Gaussian component has parameters (amplitude, mean, stddev, constant offset).
    - The fitted covariance matrix is returned internally but not used.

    Example
    -------
    >>> fit_waxs_arbitrary_n_peaks(init_guess, bounds, q, I, (100, 500), 3,
    ...                            sample_label='Fiber Sample', savefig=True)
    """
    
    x_data = xdata[q_ROI[0]:q_ROI[1]]
    y_data = ydata[q_ROI[0]:q_ROI[1]]#*1e6
    # Perform the curve fitting
    params, covariance = curve_fit(multi_gaussian, x_data, y_data, p0=initial_guess, bounds=bounds,maxfev=10000)

    y_fit = multi_gaussian(x_data, *params)
    plt.figure(figsize=(8, 6))
    plt.scatter( x_data, y_data, color='blue', label='Data', s=15)  # scatter plot for raw data
    plt.plot( x_data, y_fit, color='red', label='Fitted Gaussian')
    # Assuming params_opt contains the optimized parameters


    # Print table header
    print(f"{'Peak':<10}{'Amplitude':<15}{'Mean (q)':<15}{'Stddev (FWHM)':<20}{'C':<10}")
    print("-" * 70)


# Extract and print parameters for each Gaussian
    for i in range(n_gauss_peaks):
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]
        #print(f"Gaussian {i + 1}: Amplitude = {amp:.3e}, Mean (q) = {mean:.3e}, Stddev (FWHM) = {stddev:.3e}, c = {c:.3e}")
        #print(f"{amp:.3e}, {mean:.3e}, {stddev:.3e}, {c:.3e}")
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]

        print(f"Gaussian {i + 1:<2} {amp:<15.3e} {mean:<15.3e} {stddev:<20.3e} {c:<10.3e}")
        
        # Plot each Gaussian
        if i <= 1:
            label = f'peak {i + 1}, Kapton'
        elif i == 2:
            label = f'peak {i + 1}, background'
        else:
            label = f'peak {i + 1}, Sample'
        
        plt.plot(
            x_data, 
            amp * np.exp(-0.5 * ((x_data - mean) / stddev) ** 2) + c,
            linestyle='--',
            label=label
        )



    # Add labels and title
    plt.xlabel('q')
    plt.ylabel('I')
    
    
    # Show legend
    plt.legend()
    plt.title(sample_label)
    
    # Display the plot
    if savefig:
        if filepath is None:
            filepath = os.getcwd()  # default to current working directory
        figname = fig_name 
        
        plt.savefig(os.path.join(filepath,figname), dpi=300, bbox_inches="tight")
    
    plt.show()
    
    # Print the fitted parameters


def fit_waxs_arbitrary_n_peaks_linbg(initial_guess,bounds,xdata,ydata,q_ROI,n_gauss_peaks,peak_labels,sample_label='sample',fig_name="figure",savefig=False,filepath=None):
    """
    Fit a selected q-range of WAXS data with multiple Gaussian peaks plus a linear background.

    This version of the fitting function includes a linear background (y = m*x + c)
    in addition to the specified number of Gaussian peaks. It prints a formatted
    table of fitted parameters and plots each component along with the data.

    Parameters
    ----------
    initial_guess : list or ndarray
        Initial guess for all fitting parameters, including the linear background.
        Should contain 4 * n_gauss_peaks + 2 elements:
        [amp1, mean1, stddev1, c1, ..., ampN, meanN, stddevN, cN, m, c_bg].
    bounds : 2-tuple of array-like
        Lower and upper bounds for all parameters.
    xdata : ndarray
        Array of q-values.
    ydata : ndarray
        Array of corresponding intensity values.
    q_ROI : tuple of int
        Indices defining the region of q-values to fit, e.g. (start_idx, end_idx).
    n_gauss_peaks : int
        Number of Gaussian peaks to include in the fit.
    peak_labels : list of str
        Custom labels for each Gaussian peak to display in the legend and printed table.
    sample_label : str, optional
        Label used as the plot title. Default is 'sample'.
    fig_name : str, optional
        Name of the output figure file if saving is enabled. Default is 'figure'.
    savefig : bool, optional
        Whether to save the resulting plot as an image. Default is False.
    filepath : str or None, optional
        Directory to save the figure. If None, saves in the current working directory.

    Outputs
    -------
    - Prints a formatted table of Gaussian peak parameters and linear background coefficients.
    - Displays a plot of the data, total fit, individual Gaussians, and the background line.

    Notes
    -----
    - Uses `multi_gaussian_linbg(x, *params)` to define the full model.
    - Each Gaussian has four parameters (amplitude, mean, stddev, offset),
      and two additional parameters (m, c) describe the linear background.
    - The covariance matrix from the fit is calculated but not returned.

    Example
    -------
    >>> fit_waxs_arbitrary_n_peaks_linbg(init_guess, bounds, q, I, (200, 800), 4,
    ...                                  ['Kapton 1', 'Kapton 2', 'Background', 'Sample'],
    ...                                  sample_label='Delithiated Fiber')
    """


    x_data = xdata[q_ROI[0]:q_ROI[1]]
    y_data = ydata[q_ROI[0]:q_ROI[1]]#*1e6
    # Perform the curve fitting
    params, covariance = curve_fit(multi_gaussian_linbg, x_data, y_data, p0=initial_guess, bounds=bounds,maxfev=10000)

    y_fit = multi_gaussian_linbg(x_data, *params)
    plt.figure(figsize=(8, 6))
    plt.scatter( x_data, y_data, color='blue', label='Data', s=15)  # scatter plot for raw data
    plt.plot( x_data, y_fit, color='red', label='Fitted Gaussian')
    # Assuming params_opt contains the optimized parameters


    # Print table header
    print(sample_label)
    header = ["Peak", "Label", "Amplitude", "Mean (q)", "Stddev (FWHM)", "C"]
    print("{:<6}\t{:<25}\t{:<15}\t{:<15}\t{:<20}\t{:<10}".format(*header))
    print("-" * 100)



# Extract and print parameters for each Gaussian
    for i in range(n_gauss_peaks):
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]
        #print(f"Gaussian {i + 1}: Amplitude = {amp:.3e}, Mean (q) = {mean:.3e}, Stddev (FWHM) = {stddev:.3e}, c = {c:.3e}")
        #print(f"{amp:.3e}, {mean:.3e}, {stddev:.3e}, {c:.3e}")
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]

        # print(f"Gaussian {i + 1:<2} {amp:<15.3e} {mean:<15.3e} {stddev:<20.3e} {c:<10.3e}")
        label = peak_labels[i]
        print("{:<6}\t{:<25}\t{:<15.3e}\t{:<15.3e}\t{:<20.3e}\t{:<10.3e}".format(
        f"Gaussian {i+1}", label, amp, mean, stddev, c))
        
        # Plot each Gaussian
        # if i <= 1:
        #     label = f'peak {i + 1}, Kapton'
        # elif i == 2:
        #     label = f'peak {i + 1}, background'
        # else:
        #     label = f'peak {i + 1}, Sample'
        
        plt.plot(
            x_data, 
            amp * np.exp(-0.5 * ((x_data - mean) / stddev) ** 2) + c,
            linestyle='--',
            label=label
        )

    m,c = params[-2:]
    plt.plot(x_data, m*x_data+c,
            linestyle='--',
            label="background"
        )
    print(f"Lin BG {m:<15.3e} {c:<15.3e}")


    # Add labels and title
    plt.xlabel('q')
    plt.ylabel('I')
    #plt.title('Delith, C spacing 1-0, para')
    
    # Show legend
    plt.legend()
    plt.title(sample_label)
    
    # Display the plot
    # Save figure if requested
    if savefig:
        if filepath is None:
            filepath = os.getcwd()  # default to current working directory
        os.makedirs(filepath, exist_ok=True)  # make sure directory exists
        plt.savefig(os.path.join(filepath, fig_name), dpi=300, bbox_inches="tight")

    
    plt.show()
    
    # Print the fitted parameters



def plot_2d_scan(
    filepath,
    this_q_range_in,
    theta_ranges,
    invert_theta_mask,
    this_ROI_size,
    vmaxes,
    vmins,
    xlim,
    ylim,
    save=False,
    savepath=None,
    show=True,
):
    """
    Plot a 2D scan in reciprocal space (qx, qy) from the given data file and optionally save it.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    this_q_range_in : tuple or list
        q-range input for `get_2d`.
    theta_ranges : tuple or list
        Theta angle ranges used for extraction.
    invert_theta_mask : bool
        Whether to invert the theta mask.
    this_ROI_size : int or tuple
        Region of interest size passed to `get_2d` and `get_real_q`.
    vmaxes : float
        Maximum value for colormap scaling.
    vmins : float
        Minimum value for colormap scaling.
    xlim : tuple
        Limits for the x-axis, e.g. (-3, 3).
    ylim : tuple
        Limits for the y-axis, e.g. (-3, 3).
    save : bool, optional
        Whether to save the figure. Default is False.
    savepath : str, optional
        Path to save the plot (e.g. 'results/scan_plot.png').
        If None and `save=True`, saves to the current working directory with an autogenerated name.
    show : bool, optional
        Whether to display the plot. Defaults to True.
    """
    # Compute the 2D scan
    this_scan = get_2d(filepath, this_q_range_in, theta_ranges, invert_theta_mask, ROI_size=this_ROI_size)

    # Extract metadata and axis info
    sample_name = get_sample_name(filepath)
    q_range = get_real_q(filepath, this_ROI_size)
    axis_data_real_units = np.concatenate((-np.flip(q_range, 0), q_range))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(
        np.log(this_scan + 1e-2),
        vmin=vmins,
        vmax=vmaxes,
        cmap="turbo",
        aspect=1,
        extent=[
            axis_data_real_units[0],
            axis_data_real_units[-1],
            axis_data_real_units[0],
            axis_data_real_units[-1],
        ],
    )
    plt.title(sample_name)
    plt.xlabel(r"$q_x$ $(Å^{-1})$")
    plt.ylabel(r"$q_y$ $(Å^{-1})$")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()

    # Save if requested
    if save:
        if savepath is None:
            filename = f"{sample_name}_2Dscan.png"
            savepath = os.path.join(os.getcwd(), filename)
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to: {savepath}")

    # Show or close
    if show:
        plt.show()
    else:
        plt.close()





def fit_waxs_arbitrary_n_peaks_linbg_return_vals(initial_guess,bounds,xdata,ydata,q_ROI,n_gauss_peaks,peak_labels,sample_label='sample',fig_name="figure",savefig=False,filepath=None):
    x_data = xdata[q_ROI[0]:q_ROI[1]]
    y_data = ydata[q_ROI[0]:q_ROI[1]]#*1e6
    # Perform the curve fitting
    params, covariance = curve_fit(multi_gaussian_linbg, x_data, y_data, p0=initial_guess, bounds=bounds,maxfev=10000)

    y_fit = multi_gaussian_linbg(x_data, *params)



    # Print table header
    print(sample_label)
    print(f"{'Peak':<10}{'Amplitude':<15}{'Mean (q)':<15}{'Stddev (FWHM)':<20}{'C':<10}")
    print("-" * 70)


# Extract and print parameters for each Gaussian
    for i in range(n_gauss_peaks):
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]
       
        amp, mean, stddev, c = params[4 * i:4 * (i + 1)]

        # print(f"Gaussian {i + 1:<2} {amp:<15.3e} {mean:<15.3e} {stddev:<20.3e} {c:<10.3e}")
        print(f"Gaussian {i + 1:<2} | {amp:<15.3e} | {mean:<15.3e} | {stddev:<20.3e} | {c:<10.3e}")
        label = peak_labels[i]
        
    return params

    # Print the fitted parameters
def get_sample_name(filename):
    data = fabio.open(filename)
    #first we need to convert the existing 0-1 q to n. pixels
    sample_name = (data.header.get('Comment'))
    print("sample name:",sample_name)
    return sample_name

def get_1D(filename,q_range_in,THETA,INVERT,ROI_size=150,custom_SD=55.0004):
    data2dIMG = open_edf(filename)
    module_edges = (data2dIMG.data == -1)
    
    # ROI_size = 150
    # Trim the data
    DATA = trim_img(data2dIMG, ROI_size)
    module_edges_trim = trim_mask(data2dIMG,module_edges,ROI_size)
    # Save theta and q ranges
    # THETA = [-3.14 ,3.14]
    # INVERT = False
    # Q = [0.0, 1]
    q = get_real_q(filename,ROI_size,custom_SD)
    
    _, intens = calc_intensity_over_q(DATA,ROI_size,THETA,INVERT,q_range_in,module_edges_trim)
    

    return q,intens

    
def get_2d(filename,q_range_in,THETA,INVERT,ROI_size=150):
    data2dIMG = open_edf(filename)
    module_edges = (data2dIMG.data == -1)
    # ROI_size = 150
    # Trim the data
    DATA = trim_img(data2dIMG, ROI_size)
    module_edges_trim = trim_mask(data2dIMG,module_edges,ROI_size)
    theta_mask = calc_theta_mask(DATA,THETA, INVERT)
    q_mask = calc_q_mask(DATA,q_range_in)
    return DATA * q_mask * theta_mask *module_edges_trim 

def calc_intensity_over_q(DATA,num,THETA,INVERT,q_range_in,module_edges_trim):
    theta_mask = calc_theta_mask(DATA,THETA, INVERT)
    qs = np.linspace(q_range_in[0], q_range_in[1], num)
    intensity = np.zeros(num)
    rgrid = calc_r_grid(DATA)
    for i in range(0, num-1):
        # mask = theta_mask * calc_q_mask(DATA,qs[i:i+2])*module_edges_trim
        mask = theta_mask * calc_q_mask_precalc_rgrid(rgrid,qs[i:i+2])*module_edges_trim 
        if np.sum(mask) == 0:
            intensity[i] = np.sum(DATA * mask) / np.sum(mask+1e-9)
        else:
            intensity[i] = np.sum(DATA * mask) / np.sum(mask)
            
        

    return qs, intensity
    
def calc_mesh_grid(data_in):
    # Meshgrid
    x = np.linspace(-1, 1, data_in.shape[0])
    y = np.linspace(-1, 1, data_in.shape[1])
    X, Y = np.meshgrid(x, y)
    return X, Y

def calc_theta_mask(data_in,theta, invert=False):
    X, Y = calc_mesh_grid(data_in)
    # Polar coordinates
    theta_grid = np.arctan2(Y, X)
    # Mask
    theta_mask = (theta_grid > theta[0]) & (theta_grid < theta[1])
    if invert:
        theta_mask = np.ones(theta_grid.shape) - theta_mask
    return theta_mask

def calc_r_grid(data_in):
    X, Y = calc_mesh_grid(data_in)
    # Calculate q
    rgrid = np.sqrt(X**2 + Y**2)
    return rgrid
def calc_q_mask_precalc_rgrid(rgrid,q_range):
    q_mask = (rgrid > q_range[0]) & (rgrid < q_range[1])
    return q_mask
def calc_q_mask(data_in,q_range):
    X, Y = calc_mesh_grid(data_in)
    # Calculate q
    rgrid = np.sqrt(X**2 + Y**2)
    q_mask = (rgrid > q_range[0]) & (rgrid < q_range[1])
    return q_mask

def trim_img(img, ws):
    # Read image center coordinates from the header
    xcen = int(round(float(img.header["Center_2"])))
    ycen = int(round(float(img.header["Center_1"])))
    
    # Trim data around the center
    data = img.data[xcen-ws:xcen+ws,ycen-ws:ycen+ws]
    
    data[data<1] = 0
    return data
def trim_mask(img,mask,ws):
    # Read image center coordinates from the header
    xcen = int(round(float(img.header["Center_2"])))
    ycen = int(round(float(img.header["Center_1"])))
    
    # Trim data around the center
    mask_trim = 1 - mask[xcen-ws:xcen+ws,ycen-ws:ycen+ws]
    
    #data[data<1] = 0
    return mask_trim