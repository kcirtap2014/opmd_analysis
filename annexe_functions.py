import numpy as np
import math
import scipy.constants as cst
import random as rnd
from scipy.stats import multivariate_normal

# Constant
SENTINEL = float("inf")

class Fields():
    """
    This class prepares the Fields attributes 
    """
    def __init__(self, OPMD_obj, it):
        """
        Prepares the Fields attributes. The attributes are:
         - E-fields: Ex, Ey, Ez
         - B-fields: Bx, By, Bz
         - the longitudinal position: z
         - Shapes of the date in x and z-direction: x_shape, z-shape 

        Parameters:
        -----------
        OPMD_obj : an OpenPMD object
        it : type int. Chosen iteration
        """
        self.Ex, self.info_Ex = OPMD_obj.get_field( iteration=it,  field='E', 
            coord='x' )
        self.Ey, self.info_Ey = OPMD_obj.get_field( iteration=it,  field='E', 
            coord='y' )
        self.Ez, self.info_Ez = OPMD_obj.get_field( iteration=it,  field='E', 
            coord='z' )
        self.Bx, self.info_Bx = OPMD_obj.get_field( iteration=it,  field='B', 
            coord='x' )
        self.By, self.info_By = OPMD_obj.get_field( iteration=it,  field='B', 
            coord='y' )
        self.Bz, self.info_Bz = OPMD_obj.get_field( iteration=it,  field='B', 
            coord='z' )
        self.z = self.info_Ez.z

        # Shape in transverse and longitudinal direction
        self.x_shape, self.z_shape = np.shape(self.Ex) 

    def maxfield(self):
        """
        Return the position of max laser field starting from z>0

        Return
        ------
        self.z[index]: the longitudinal position at max Ey field
        """
        index = np.argmax(self.Ey[self.x_shape/2,:])

        return self.z[index]

    def bucket (self):
        """
        Returns the boundaries of the buckets. The buckets are determined by 
        a change of sign in Ez-field

        Return
        ------
        buckets : 2D array. 
            Ex: buckets[i] gives you the ith bucket, i=0 signifies
                the first bucket. 
                buckets[i][j], j represents either the minimum bound or the 
                maximum bound of the ith bucket
        """
        max_laser_z = self.maxfield()
        z_array = np.arange(self.z_shape)
        index = np.compress(self.z <= max_laser_z, z_array)
        
        # Savitzky-golay filtering is applied to smooth the Ez-field in order 
        # to avoid any abrupt change of sign due to the noise
        ez_filtered = savitzky_golay(self.Ez[self.x_shape/2,:], 51, 3)
        ez_temp = np.take(ez_filtered,index)
        z_temp = np.take(self.z,index)
        root_zero = find_root(ez_temp,z_temp)  

        # Length of root_zero
        lrz = np.shape(root_zero)[0]

        k=-1
        j=0

        buckets=[[] for i in xrange(lrz/2)]
        
        for i in range(lrz-1,-1,-1):

            if (j%2)==0:
                j=0
                k+=1
                # we want the minimum bound to be on the right
                i-=1
            else:
                i+=1
            buckets[k].append(root_zero[i])
            j+=1
        
        return buckets

class Particles():
    """
        This class defines all the attributes related to the particles
    """
    def __init__(self, OPMD_obj, it, species="electrons"):
        """
        Contains the following particle quantities:
        - particle positions in 3 dimensions : x, y, z
        - particle velocities in 3 dimensions : ux, uy, uz
        - particle weight : w
        - particle gamma : gamma, computed by the formula
            sqrt(1 + (ux^2 + uy^2 + uz^2)/c^2))
        
        Parameters:
        ----------
        OPMD_obj : OpenPMD object
        it : chosen iteration
            -int
        species : requested Species 
            -String 
        """
        v_list = OPMD_obj.get_particle(
            var_list=['x','y','z','ux','uy','uz','w'], iteration=it, 
            species=species) 
        self.x = v_list[0]
        self.y = v_list[1]
        self.z = v_list[2]
        self.ux = v_list[3]
        self.uy = v_list[4]
        self.uz = v_list[5]
        self.w = v_list[6]
        self.gamma = np.sqrt(1 + (self.ux**2 + self.uy**2 + self.uz**2))

    def filter(self, gamma_threshold=[], ROI=[]): 
        """
        Returns the filtered quantities based on the selection criteria

        Paramters:
        ----------
        gamma_threshold: selection in gamma
            - array of floats
        ROI: selection in space
            - array of floats

        Returns:
        --------
        Index: an array of indices which satisfy the conditions
            -array of ints
        """
        if not gamma_threshold:
            gamma_threshold.append(0.)
            
        if np.shape(gamma_threshold)[0]==1:
            gamma_threshold.append(SENTINEL)

        z_array = np.arange(np.shape(self.z)[0])

        # We impose the use of the unit micron
        ROI = np.array(ROI)*1e6
        if np.array(ROI).size==0:
            index = np.compress(np.logical_and(self.gamma>=gamma_threshold[0], \
                self.gamma<=gamma_threshold[1]), z_array)
            print "No filtering in z"
        else:
            index = np.compress(np.logical_and(np.logical_and(
                self.gamma>=gamma_threshold[0], self.gamma<=gamma_threshold[1]), 
            np.logical_and(self.z>=ROI[0], self.z<=ROI[1])), z_array)
        return index

def normalize_data(x_beam, ux_beam):
    """
    Returns the normalized data for Machine learning algorithm
    The reason for normalizing is that the dynamics of the features are
    very different.

    X=(X-mean(X))/dynamics(X)

    Parameters
    ------
    x_beam ,ux_beam

    Returns
    -------
    Normalized data, X dimension m*n
    m = number of data
    n = number of features
    """
    x_beam_norm = (x_beam-np.mean(x_beam))/(max(x_beam)-min(x_beam))
    ux_beam_norm = (ux_beam-np.mean(ux_beam))/(max(ux_beam)-min(ux_beam))
    X = np.transpose(np.vstack((x_beam_norm,ux_beam_norm)))
    return X

def denormalize_data( x, y, x_beam, y_beam ):
    """
    Returns the denormalized data for Machine learning algorithm
    The reason for normalizing is that the dynamics of the features are
    very different.

    X=(X-mean(X))/dynamics(X)

    Parameters
    ------
    x, y, x_beam ,ux_beam

    Returns
    -------
    Deormalized data, new_x, new_y
    
    """
    new_x = x*(max(x_beam) - min(x_beam)) + np.mean(x_beam)
    new_y = y*(max(y_beam) - min(y_beam)) + np.mean(y_beam)

    return new_x, new_y

def cross_validate_data(X_val,sigma_factor=1.):
    """
    Returns y_val which tells anomalous particles in the cross-validated data set

    Parameters
    ------
    X_val: cross validated data set
    sigma_factor: factor that is multiplied to the standard
                  deviation 

    Returns
    -------
    y_val=1 if partcles are anomalous

    """
    if sigma_factor == 0:
         y_val = np.ones(np.shape(X_val))
    else:
        y_val = np.zeros(np.shape(X_val))
        x = np.arange(np.shape(X_val[:,0])[0])
        x_sigma = np.std(X_val[:,0])
        x_mean = np.mean(X_val[:,0])
        s_sigma = sigma_factor*x_sigma
        ii = np.compress(((X_val[:,0]<x_mean-s_sigma) | (X_val[:,0]>x_mean+s_sigma)), x)
        y_val[ii,:] = 1
    return y_val

def estimateGaussian(X):
    """
    Returns mean and covariance of the dataset

    Parameters
    ------
    X : data set

    Returns
    -------
    mu : mean of the dataset
    sigma2 : covariance of the dataset
    """

    mu = np.mean(X,axis=0)
    sigma2 = np.cov(np.transpose(X))

    return mu, sigma2

def multivariateGaussianDistrib(mu, sigma2):
    """
    Generate a multivariate Gaussian Distribution samples

    Parameters:
    -----------
    mu : calculated mean
    sigma : calculated standard deviation

    Returns:
    --------
    x,y : samples
    """
    x,y = rnd.multivariate_normal(mu, sigma2, 50000).T

    return x,y
    
def selectThreshold( y_val, p_val ):
    """
    Returns best F1 score and epsilon
    do a scan of the epsilon and calculate the corresponding F1 score to determine
    the most suitable epsilon

    precision =  true_positive/(true_positive+false_positive)
    recall    =  true_positive/(true_positive+false_negative)
    F1 score  =  2*precision*recall/(precision+recall)

    Parameters
    ----------
    y_val = 0 or 1 if 1: particles are anomalous
    p_val = probability distribution of the cross-validated dataset

    Returns
    -------
    best_f1 : best F1 score      
    best_epsilon : we then apply this epsilon for prediction

    """
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    #step_size = (max(p_val) - min(p_val)) / 1e3
    step_range=[]
    x=min(p_val)
    
    while(x<max(p_val)):
        x*=2
        step_range.append(x)
    

    for epsilon in step_range:#arange(min(p_val), max(p_val)+step_size, step_size ):
        
        prediction = (p_val < epsilon).reshape( np.shape( y_val)[0], 1 )

        true_pos  = np.sum((prediction == 1) & (y_val == 1))
        false_pos = np.sum((prediction == 1) & (y_val == 0))
        false_neg = np.sum((prediction == 0) & (y_val == 1))

        precision   = true_pos * 1.0 / (true_pos + false_pos)
        recall      = true_pos * 1.0 / (true_pos + false_neg)

        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1         = f1
            best_epsilon    = epsilon

    return best_f1, best_epsilon

def multivariateGaussian( X, mu, sigma2, n=2 ):
    """
    Returns the probability of the dataset following a multivariate 
    Gaussian distribution

    Parameters
    ------
    X  = dataset
    mu = mean of the dataset
    sigma2 = covariance of the dataset
    n  = number of features (normally 2)

    Returns
    -------
    p = probability of the dataset
    """
    temp_X = X - mu
    p = (2.0*math.pi)**(-n/2.0)*np.linalg.det(sigma2)*np.exp(-.5*\
        np.sum((temp_X*np.diagonal(np.linalg.pinv(sigma2)))*temp_X,axis=1))
    
    return p

def run_anomaly_detection(training_data, real_data, n=2, l_visualize_particles=False, sigma_factor=1):
    """
    Performs anomaly detection and returns the indices of the normal data
    
    Parameters
    ------
    training_data : a dictionary of x_beam, ux_beam
    real_data : a dictionary of x_beam, ux_beam
    n : number of features

    Returns
    -------
    chosen : index of selected particles
    """

    x_beam_training = training_data['x_beam']
    ux_beam_training = training_data['ux_beam']
    x_beam = real_data['x_beam']
    ux_beam = real_data['ux_beam']

    ##--preparing the real data set
    X = normalize_data(x_beam, ux_beam)
    ##--the cross validated data set

    i_cv = rnd.sample(range(0,len(ux_beam_training)), int(0.4*len(ux_beam_training)))

    ux_beam_cv = np.take(ux_beam_training,i_cv)
    x_beam_cv  = np.take(x_beam_training,i_cv)

    ##--normalize the cross validate data set
    X_val = normalize_data(x_beam_cv,ux_beam_cv)

    ##--prepare the outcome of the cross validated data set
    y_val = cross_validate_data(X_val,sigma_factor)

    ##--normalize the training and cross-validated data set 
    X_training = normalize_data(x_beam_training, ux_beam_training)

    ##--estimate the Gaussian parameters from the training set
    mu,sigma2 = estimateGaussian(X_training)

    ##--probabilty of the real data set
    p = multivariateGaussian(X,mu, sigma2, n)

    ##--estimate the probability of the cross-validate data set 
    ##with the training set statistical parameters
    p_val = multivariateGaussian(X_val,mu, sigma2, n)

    ##-- determining the best epsilon values
    best_f1, epsilon = selectThreshold( y_val, p_val )

    ##--to see how good is your prediction
    print "Performance score of the anomaly detection algorithm:\n"
    print "F1 score: " ,best_f1 
    print "best epsilon: ",epsilon 

    ##--eliminate the outliers
    chosen = np.compress(p>epsilon, np.arange(np.shape(p)[0]))

    return chosen 
        
def emittance_calc( x, ux, w):
    """
    Calculation on emittance based on statistical approach in J. Buon (LAL) 
    Beam phase space and Emittance. We first calculate the covariance, and 
    the emittance is epsilon=sqrt(det(covariance matrix)).
    Covariance is calculated with the weighted variance based on 
    http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf 

    Parameters:
    -----------
    x : x of the beam
        - array of floats 
    ux : ux of the beam
        - array of floats 
    w : weight of the beam
        - array of floats 

    Returns:
    --------
    emittance_w : emittance
        - float
    """
    try:
        w_x = np.mean(x)
        w_ux = np.mean(ux)
        z_array = np.arange(np.shape(x)[0])
        selected_array = np.compress(w != 0., z_array )
        nz_w = np.take(w, selected_array)
        xux = np.sum(x*ux)/(np.shape(x)[0])
        variance_x = np.var(x)
        variance_ux = np.var(ux)
        covariance_xux = xux - w_x*w_ux
        xuxw = [[variance_x,covariance_xux],[covariance_xux,variance_ux]]

        emittance_w = np.sqrt(np.linalg.det(xuxw))

        if math.isnan(emittance_w):
            emittance_w=0.0

    except ZeroDivisionError:
        emittance_w=0
  
    return emittance_w

def gamma2energy(gamma_beam):
    """
    Converts gamma to energy: energy = gamma*0.511 [MeV]

    Parameters:
    -----------
    gamma_beam : beam gamma
        - array of floats

    Returns:
    --------
    energy : beam energy
    """
    try:
        energy=[i*0.511 for i in gamma_beam]
    except TypeError:
        energy=0

    return energy

def beam_charge(z_beam, w_beam, weight_species):
    """
    Compute the charge 

    Parameters:
    -----------
    z_beam : beam z
        - array of floats
    w_beam : beam weight
        - array of floats
    weight_species : weight of the species
        - a float value

    Returns:
    --------
    energy : charge of the beam
    """
    try:
        charge = np.sum(w_beam)*cst.e*1e-9*weight_species
    except TypeError:
        charge = 0.
        print "No particles are detected"

    return charge

def beam_numParticles(z_beam):
    """
    Compute the number of particles

    Parameters:
    -----------
    z_beam : beam z
        - array of floats

    Returns:
    --------
    numPart : number of particles
    """
    try:
        numPart= np.shape(z_beam)[0]
    except TypeError:
        numPart = 0.
        print "No particles are detected"
        
    return numPart

def beam_energy(gamma_beam,z_beam,w_beam):
    """

    """
    energy = gamma2energy(gamma_beam)
    try:
        n_energy,energy = np.histogram(energy,bins=100,weights=w_beam)
        n_energy*=cst.e*10**6  ##for MeV
        energy = np.delete(energy,0)
    except TypeError:
        energy = 0.
        n_energy = 0.
        print "No particles are detected"

    return n_energy, energy

def beam_statistics(gamma_beam,z_beam,w_beam):   
    n_energy,energy=beam_energy(gamma_beam,z_beam,w_beam)
    try:
        average_energy = np.average(energy, weights = n_energy)
        variance = np.average((energy - average_energy)**2, weights=n_energy)
        std = np.sqrt(variance)
        eSpread = std/average_energy
    except ZeroDivisionError:
        average_energy = 0
        eSpread = 0
    #energy_spread = delta_EsE(energy,n_energy,l_fwhm) 
    return average_energy, eSpread

def beam_variables(F, P, gamma_threshold=[], bucket=False, override=[]):
    """
    Generates a multi-dimensional array of particles' quantities that 
    satisfy the selection criteria

    Parameters:
    -----------
    F : Fields object, Instance of the field

    P : Paricles object, Instance of the particle

    gamma_threshold: an array of floats, selection in gamma 

    bucket : an array of floats, selection in space
        - Corresponding values:
            -- bucket==0 : No bucket
            -- bucket==1 : bucket in the decelerating field
            -- bucket==2 : bucket in the accelerating field

    override: an array of floats, selection in space given 
        directly by the user

    Returns:
    --------
    selected_beam : an array of selected particles' quantities
        - shape (8, num_selected_particles,)
        - Corresponding indices:
            -- selected_beam[0]: x
            -- selected_beam[1]: y
            -- selected_beam[2]: z
            -- selected_beam[3]: ux
            -- selected_beam[4]: uy
            -- selected_beam[5]: uz
            -- selected_beam[6]: w
            -- selected_beam[7]: gamma
    """
    if bucket:
        buckets= F.bucket()
        if bucket == 1:
            buckets=[buckets[0][1],max(F.z)]
        if bucket == 2:
            buckets = buckets[0]
    else:   
        buckets=[]
    if override:
        buckets = override+[max(F.z)]
    
    index = P.filter(gamma_threshold, buckets)
    x_beam = P.x[index]
    y_beam = P.y[index]
    z_beam = P.z[index]
    ux_beam = P.ux[index]
    uy_beam = P.uy[index]
    uz_beam = P.uz[index]
    w_beam = P.w[index]
    gamma_beam = P.gamma[index]

    selected_beam = np.vstack((x_beam, y_beam, z_beam, ux_beam, 
        uy_beam, uz_beam, w_beam, gamma_beam))

    return selected_beam

def find_root(ez,z):

    #displaying the sign of the value
    l = np.shape(ez)[0]
    s = np.sign(ez)
    index = []

    for i in range (0,l-1):
        if (s[i+1]+s[i] ==0 ):
            # when the sum of the signs ==0, that means we hit a 0
            index.append(i)
    
    root_zero = np.take(z,index).tolist()
    lrz = len(root_zero)
    # if there's only one root found, 
    # there should be an end to it, we consider the min z
    # as the the limit
        
    # insert a z value at the first index
    if lrz==1:
        root_zero.insert(0,min(z))
    # if length of root is not pair, we remove the first value
    if np.shape(root_zero)[0]%2 != 0:
        root_zero = np.delete(root_zero,0)
    return root_zero

def get_rms(variable):
    """
    Returns the rms values of a particle quantity

    Parameters:
    -----------
    variable : the quantity that we want to perform standard deviation
        calculation
        - an array of floats

    Returns:
    --------
    rms : rms value of the quantity
    """
    rms = np.std(variable)
    if math.isnan(rms):
        rms = 0
    return rms

def vz_measurement(z_prev, z_cur, t_diff):
    """
    Returns the vz of the plasma wave

    Parameters:
    -----------
    z_prev : the previous instant of zero-crossing position 
        - float value

    z_cur : the current instant of zero-crossing position
        - float value

    t_diff : time difference between these two instants
        - float value

    Returns:
    --------
    vz : the velocity of the plasma wave 
    """

    return (z_cur - z_prev)/t_diff

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """

    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * math.factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
