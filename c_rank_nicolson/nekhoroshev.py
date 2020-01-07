import numpy as np
import scipy as sc
from tqdm import tqdm
import scipy.integrate as integrate

from .c_rank_nicolson import crank_nicolson

# These three functions are necessary for the fourth one.

def nek_D(I, c, I_star, alpha, beta):
    return(
        c
        * np.power(I, beta)
        * np.exp(- 2 * np.power(I_star / I, alpha))
    )


def x_0(I_0, I_max, c, I_star, alpha, beta):
    return integrate.quad(
        lambda x: x / np.sqrt(nek_D(x, c, I_star, alpha, beta)),
        I_0, I_max
    )[0]


def t_max_approx(I_0, I_max, c, I_star, alpha, beta):
    x0 = x_0(I_0, I_max, c, I_star, alpha, beta)
    return x0 * x0 / 3


def dt_approx(I_0, I_max, c, I_star, alpha, beta):
    """Get a valid approximation of dt for observing "properly" the peak of
    a current (based on Rocco's calculations).
    
    Parameters
    ----------
    I_0 : float
        starting point of the delta distribution we want to observe
    I_max : float
        absorbing barrier position
    c : float
        scaling constant
    I_star : float
        Nekhoroshev constant
    alpha : float
        Nekhoroshev exponent
    beta : float
        polynomial exponent
    
    Returns
    -------
    float
        the good time scaling dt for default current iteration parameters
    """
    return t_max_approx(I_0, I_max, c, I_star, alpha, beta) * 2 / 100000


def logistic_damping(I_linspace, I0, damping_point, lenght):
    """Performs a logistic damping on the given point
    of a particle distribution.
    
    Parameters
    ----------
    I_linspace : numpy 1D array
        Coordinates on which you want to work
    I0 : numpy 1D array
        Distribution on which you perform the damping
    damping_point : float
        Corresponding point on which you perform the damping
    lenght : float
        Lenght of the damping (damping point is in the middle)
    
    Returns
    -------
    numpy 1D array
        New damped distribution
    """
    I1 = I0 * (1 - 1 / (1 + 1/np.exp((I_linspace - damping_point)/lenght)))
    return I1


def gaussian_distribution(I_linspace, sigma):
    """Generates an initial gaussian distribution on the given I_linspace
    taken as reference.
    
    Parameters
    ----------
    I_linspace : numpy 1D array
        I coordinates array on which you build the gaussian distribution
    sigma : float
        Standard deviation of the distribution
    
    Returns
    -------
    numpy 1D array
        Corresponding initial condition on the array
    """
    return np.exp(- np.power(I_linspace, 2.0) / (2 * sigma * sigma)) / np.sqrt(2.0 * np.pi * sigma * sigma)


def exponential_distribution(I_linspace, beta):
    """Generates an initial exponential distribution on the given I_linspace
    taken as reference.

    Parameters
    ----------
    I_linspace : numpy 1D array
        I coordinates array on which you build the gaussian distribution
    beta : float
        Standard deviation of the distribution

    Returns
    -------
    numpy 1D array
        Corresponding initial condition on the array
    """
    return (np.exp(- I_linspace / beta) / beta)


def linear_distribution(I_linspace, max_value, stopping=-1.0):
    """Generates a linear distribution on the given I_linspace taken as reference.
    
    Parameters
    ----------
    I_linspace : numpy 1D array
        I coordinates array on which you build the linear distribution
    max_value : float
        max starting value of the distribution
    stopping : float, optional
        where to stop with the linear distribution (if default, it will consider all the space), by default -1.0
    
    Returns
    -------
    numpy array
        the linear distribution
    """    
    index = np.argmax(I_linspace >= stopping)
    return np.concatenate((np.linspace(max_value, 0.0, index), np.zeros(len(I_linspace) - index)))


def average_first_passage_time(start, stop, c, I_star, alpha, beta):
    """Estimate average first passage time
    
    Parameters
    ----------
    start : float
        starting point
    stop : float
        stopping point
    c : float
        c value
    I_star : float
        ipse dixit
    alpha : float
        nek exponential
    beta : float
        polynomial exponential
    
    Returns
    -------
    float
        average first passage time
    """    
    return 2.0 * integrate.quad(lambda x: x / nek_D(x, c, I_star, alpha, beta), start, stop)[0]


class cn_nekhoroshev(object):
    """Wrapper for the crank-nicolson method in order to do easy analysis on a Nekhoroshev-like diffusive process.
    """
    def __init__(self, I_max, c, I_star, alpha, beta, I0, dt, normalize=True, I_min=0.0):
        """Build the CN integrator for a Nekhoroshev-like diffusion process.
        
        Parameters
        ----------
        object : self
            self
        I_max : float
            Absorbing barrier position
        c : float
            Scaling constant
        I_star : float
            Nekhoroshev parameter
        alpha : float
            Nekhoroshev exponent
        beta : float
            Polynomial exponent
        I0 : numpy 1D array
            Initial distribution
        dt : float
            Time step
        normalize : bool, optional
            Do you want to normalize the initial distribution? By default True
        I_min : float, optional
            Starting point for the distribution. By default 0.0 
        
        Returns
        -------
        Object
            object
        """
        self.c = c
        self.I_star = I_star
        self.alpha = alpha
        self.beta = beta

        self.I0 = I0
        self.I_max = I_max
        self.I_min = I_min
        self.dt = dt
        self.samples = len(I0)

        self.I = np.linspace(I_min, I_max, self.samples)
        self.dI = np.absolute(self.I[1] - self.I[0])
        self.half_dI = self.dI * 0.5

        self.locked_left = False
        self.locked_right = False

        A = []
        for i in self.I:
            A.append(self.nek_D(i - self.half_dI)) if (i -
                                                       self.half_dI > 0) else A.append(0.0)
            A.append(self.nek_D(i + self.half_dI))
        A = np.array(A)
        B = np.zeros(self.samples)
        C = np.zeros(self.samples)
        D = np.zeros(self.samples + 2)

        # For Reference:
        self.diffusion = self.nek_D(self.I)

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

    def nek_D(self, x):
        """Computes Nekhoroshev diffusion coefficient for given point
        
        Parameters
        ----------
        x : float
            point of computation

        Returns
        -------
        float
            diffusion coefficient
        """
        return(
            0.5 * self.c
            * np.power(x, self.beta)
            * np.exp(- 2 * np.power(self.I_star / x, self.alpha))
        )

    def extend(self, I_max):
        """Extends the size of the simulation and moves forward the absorbing
        barrier.
        
        Parameters
        ----------
        I_max : float
            new position for absorbing barrier
        """
        assert self.I_max < I_max
        extension = np.arange(self.I_max, I_max, self.dI)[1:]
        extension = np.append(extension, extension[-1] + self.dI)

        self.I = np.append(self.I, extension)
        self.I_max = self.I[-1]

        increasing = len(self.I) - self.samples
        self.samples = len(self.I)

        self.I0 = np.append(self.I0, np.zeros(increasing))
        data = np.append(np.array(self.engine.x()), np.zeros(increasing))
        A = []
        for i in self.I:
            A.append(self.nek_D(i - self.half_dI)) if (i -
                                                       self.half_dI > 0) else A.append(0.0)
            A.append(self.nek_D(i + self.half_dI))
        A = np.array(A)
        B = np.zeros(self.samples)
        C = np.zeros(self.samples)
        D = np.zeros(self.samples + 2)

        N = self.engine.executed_iterations()
        self.engine = crank_nicolson(
            self.samples, self.I_min, self.I[-1], self.dt, data, A, B, C, D)
        if self.locked_left:
            self.engine.lock_left()
        if self.locked_right:
            self.engine.lock_right()

        self.engine.set_executed_iterations(N)

        self.diffusion = self.nek_D(self.I)

    def cut(self, damping_point, length, I_max=-1.0):
        """Executes a damping of the distribution and, if required, reduces
        the size of the simulation.
        
        Parameters
        ----------
        damping_point : float
            point of damping
        length : float
            lenght of the damping
        I_max : float, optional
            new position for the absorbing barrier, by default -1.0
        """
        assert self.I_max > I_max
        assert self.I_max > damping_point

        # Logistic damping at given point
        new_data = logistic_damping(
            self.I, self.engine.x(), damping_point, length)

        # Are we moving the collimator?
        if I_max != -1.0:
            assert damping_point <= I_max
            new_I = np.arange(self.I_min, I_max, self.dI)
            new_I = np.append(new_I, new_I[-1] + self.dI)

            self.I = new_I
            self.I_max = self.I[-1]

            self.I0 = self.I0[:len(self.I)]
            self.samples = len(self.I)
            new_data = new_data[:len(self.I)]

        # Let's make everything again
        A = []
        for i in self.I:
            A.append(self.nek_D(i - self.half_dI)) if (i -
                                                       self.half_dI > 0) else A.append(0.0)
            A.append(self.nek_D(i + self.half_dI))
        A = np.array(A)
        B = np.zeros(self.samples)
        C = np.zeros(self.samples)
        D = np.zeros(self.samples + 2)

        N = self.engine.executed_iterations()
        self.engine = crank_nicolson(
            self.samples, self.I[0], self.I[-1], self.dt, new_data, A, B, C, D)
        if self.locked_left:
            self.engine.lock_left()
        if self.locked_right:
            self.engine.lock_right()

        self.engine.set_executed_iterations(N)

        self.diffusion = self.nek_D(self.I)

    def change_dt(self, new_dt):
        """Change the dt of the integrator.
        
        Parameters
        ----------
        new_dt : float
            New value for dt
        """
        data = np.array(self.engine.x())
        self.dt = new_dt

        A = []
        for i in self.I:
            A.append(self.nek_D(i - self.half_dI)) if (i -
                                                       self.half_dI > 0) else A.append(0.0)
            A.append(self.nek_D(i + self.half_dI))
        A = np.array(A)
        B = np.zeros(self.samples)
        C = np.zeros(self.samples)
        D = np.zeros(self.samples + 2)

        N = self.engine.executed_iterations()
        self.engine = crank_nicolson(
            self.samples, self.I[0], self.I[-1], self.dt, data, A, B, C, D)
        if self.locked_left:
            self.engine.lock_left()
        if self.locked_right:
            self.engine.lock_right()

        self.engine.set_executed_iterations(N)

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
