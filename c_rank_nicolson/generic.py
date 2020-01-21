import numpy as np
import scipy as sc
from tqdm import tqdm
import scipy.integrate as integrate

from .c_rank_nicolson import crank_nicolson

# Useful functions for... our situation


def action(x, p):
    """Returns action variable
    
    Parameters
    ----------
    x : ndarray
        position
    p : ndarray
        momentum
    
    Returns
    -------
    ndarray
        action
    """
    return ((p * p) + (x * x)) * 0.5


def normed_normal_distribution(I, mean_I, sigma_I):
    """Given an I value, returns the corresponding value for a normal distribution.
    
    Parameters
    ----------
    I : ndarray
        sample points
    mean_I : float
        mean of the distribution
    sigma_I : float
        sigme of the distribution
    
    Returns
    -------
    ndarray
        results of the samples
    """
    return ((1 / np.sqrt(2 * np.pi * sigma_I ** 2))
            * np.exp(-(I - mean_I) ** 2 / (2 * sigma_I ** 2)))


def normed_normal_linspace(I_min, I_max, mean_I, sigma_I, num=100):
    """Returns a normalized linspace of a normal distribution.
    
    Parameters
    ----------
    I_min : float
        starting point
    I_max : float
        stopping point
    mean_I : float
        mean of the distribution
    sigma_I : float
        sigma of the distribution
    num : int, optional
        number of samples, by default 100
    
    Returns
    -------
    ndarray
        linspace distribution
    """
    I_list = np.linspace(I_min, I_max, num)
    values = np.empty((num))
    for i, I in enumerate(I_list):
        values[i] = normed_normal_distribution(I, mean_I, sigma_I)
    normalization = integrate.simps(values, I_list)
    values /= normalization
    return values


def x_from_I_th(I, th=np.pi / 2):
    """Returns x from action-angle variables.
    
    Parameters
    ----------
    I : float
        action value
    th : float, optional
        angle value, by default np.pi/2
    
    Returns
    -------
    float
        x value
    """
    return np.sqrt(2 * I) * np.sin(th)


def D_calculator(I, epsilon, x_star, delta, exponent):
    """Estimates D value by using definitions given for stochastic map.
    
    Parameters
    ----------
    I : float
        sampling point
    epsilon : float
        noise coefficient
    x_star : float
        nek parameter
    delta : float
        nek parameter
    exponent : float
        nek parameter (alpha)
    
    Returns
    -------
    float
        diffusion value
    """
    if I <= 0:
        return 0.0
    int_result = integrate.quad(
        (lambda th:
         epsilon ** 2
            * (2 * I)
            * np.cos(th) ** 2
            * np.exp(-np.power(((x_star) / (delta + np.absolute(x_from_I_th(I, th)))), exponent)) ** 2),
        0,
        np.pi / 2)
    # Check if int_result is valid, otherwise return 0.0
    return (int_result[0] / (np.pi / 2)
            if np.absolute(int_result[0]) > int_result[1] else 0.0)


def I_norm_sampling_to_x(mean_I, sigma_I):
    """Extracts a random action value from a normal distribution and returns a corrispective x value (assumes p=0).
    
    Parameters
    ----------
    mean_I : float
        mean of the distribution
    sigma_I : float
        sigma of the distribution
    
    Returns
    -------
    float
        extracted x
    """
    counter = 0
    while True:
        extracted_I = np.random.normal(mean_I, sigma_I)
        if extracted_I >= 0:
            break
        counter += 1
        assert counter < 100
    return x_from_I_th(extracted_I)


# The actual class to be used

class cn_generic(object):
    """wrapper for generic diffusive process"""
    def __init__(self, I_min, I_max, I0, dt, D_lambda, normalize=True):
        """init the wrapper
        
        Parameters
        ----------
        object : self
            self
        I_min : float
            starting point
        I_max : float
            absorbing point
        I0 : ndarray
            initial distribution
        dt : float
            time delta
        D_lambda : lambda
            lambda that takes an action value and returns the diffusion value
        normalize : bool, optional
            do you want to normalize the initial distribution? by default True
        """        
        self.I_min = I_min
        self.I_max = I_max
        self.I0 = I0
        self.dt = dt
        self.D_lambda = D_lambda

        self.I = np.linspace(I_min, I_max, I0.size)
        self.samples = I0.size
        self.dI = self.I[1] - self.I[0]
        self.half_dI = self.dI * 0.5

        A = []
        for i in self.I:
            A.append(self.D_lambda(i - self.half_dI)) if (i -
                                                       self.half_dI > 0) else A.append(0.0)
            A.append(self.D_lambda(i + self.half_dI))
        A = np.array(A)
        B = np.zeros(self.samples)
        C = np.zeros(self.samples)
        D = np.zeros(self.samples + 2)

        self.locked_left = False
        self.locked_right = False

        # For Reference:
        self.diffusion = np.array([self.D_lambda(i) for i in self.I])

        # Normalize?
        if normalize:
            self.I0 /= integrate.trapz(self.I0, x=self.I)

        self.engine = crank_nicolson(
            self.samples, I_min, I_max, self.dt, self.I0, A, B, C, D)

    def set_source(self, source):
        """Apply a source vector to the simulation, this will overwrite all non zero values over the simulation distribution at each iteration.
        
        Parameters
        ----------
        source : ndarray
            source to apply
        """
        self.engine.set_source(source)

    def remove_source(self):
        """Remove the source vector to the simulation.
        """        
        self.engine.remove_source()

    def lock_left(self):
        """Lock the left boundary to the non-zero value it has right now.
        """        
        self.engine.lock_left()
        self.locked_left = True

    def lock_right(self):
        """Lock the right boundary to the non-zero value it has right now.
        """
        self.engine.lock_right()
        self.locked_right = True

    def unlock_left(self):
        """Unlock the left boundary and set it to zero.
        """        
        self.engine.unlock_left()
        self.locked_left = False

    def unlock_right(self):
        """Unlock the right boundary and set it to zero.
        """
        self.engine.unlock_right()
        self.locked_right = False

    def iterate(self, n_iterations):
        """Iterates the simulation.
        
        Parameters
        ----------
        n_iterations : int
            number of iterations to perform
        """
        self.engine.iterate(n_iterations)

    def reset(self):
        """Resets the simulation to the starting condition.
        """
        self.engine.reset()

    def get_data(self):
        """Get raw distribution data.
        
        Returns
        -------
        numpy 1D array
            raw distribution data
        """
        return np.array(self.engine.x())

    def get_plot_data(self):
        """Get raw distribution data and corrispective I_linspace
        
        Returns
        -------
        (numpy 1D array, numpy 1D array)
            (I_linspace, distribution data)
        """
        return (self.I, np.array(self.engine.x()))

    def get_sum(self):
        """Get integral of the distribution (i.e. number of particles)
        
        Returns
        -------
        float
            Number of particles
        """
        return integrate.trapz(self.engine.x(), x=self.I)

    def get_particle_loss(self):
        """Get amount of particle loss (when compared to starting condition)
        
        Returns
        -------
        float
            Particle loss quota
        """
        return -(
            integrate.trapz(self.get_data(), x=self.I) -
            integrate.trapz(self.I0, x=self.I)
        )

    def current(self, samples=5000, it_per_sample=20, disable_tqdm=True):
        """Perform automatic iteration of the simulation 
        and compute resulting current.
        
        Parameters
        ----------
        samples : int, optional
            number of current samples, by default 5000
        it_per_sample : int, optional
            number of sim. iterations per current sample, by default 20
        
        Returns
        -------
        (numpy 1D array, numpy 1D array)
            (times of the samples, current values for those samples)
        """
        current_array = np.empty(samples)
        temp1 = self.get_sum()
        times = (np.arange(samples) * it_per_sample +
                 self.engine.executed_iterations()) * self.dt
        for i in tqdm(range(samples), disable=disable_tqdm):
            self.engine.iterate(it_per_sample)
            temp2 = self.get_sum()
            current_array[i] = (temp1 - temp2) / self.dt
            temp1 = temp2
        return times, current_array
