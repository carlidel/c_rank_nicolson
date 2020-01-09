import numpy as np
import scipy as sc
from tqdm import tqdm
import scipy.integrate as integrate

from .c_rank_nicolson import crank_nicolson


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

        self.I = np.linspace(I_min, I_max, I0.size())
        self.samples = I0.size()
        self.dt = self.I[1] - self.I[0]
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
        self.diffusion = self.D_lambda(self.I)

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