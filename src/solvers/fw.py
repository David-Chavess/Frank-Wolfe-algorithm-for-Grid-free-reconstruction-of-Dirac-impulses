import datetime as dt
from time import time

import numpy as np
import pyxu.abc.operator as pxo
import pyxu.abc.solver as pxs
import pyxu.info.ptype as pxt
import pyxu.operator as pxop
import pyxu.opt.stop as pxos
import pyxu.runtime as pxrt
import pyxu.util as pxu
from pyxu.abc import StoppingCriterion
from pyxu.opt.solver.pgd import PGD

from src.operators.fourier_operator import DualCertificate, MyLinOp, TMP


class FW(pxs.Solver):

    def __init__(
            self,
            measurements: pxt.NDArray,
            forward_op: MyLinOp,
            lambda_: float,
            **kwargs):
        super().__init__(**kwargs)

        self.y = measurements
        self.forward_op = forward_op
        self.lambda_ = lambda_
        self.x_dim = 1

        # Initialize the swarm parameters
        self.swarm_bounds = np.array([[0], [1]])
        self.swarm_iterations = 10
        self.swarm_n_particles = 100
        self.swarm_w = 0.25
        self.swarm_c1 = 0.75
        self.swarm_c2 = 0.25

        # Initialize the correction parameters
        self._min_correction_steps = 100
        self._max_correction_steps = 100
        self._init_correction_prec: float = 0.2
        self._final_correction_prec: float = 1e-4

        self.amplitude_threshold = 0.01

        self.verbose = True
        self.show_progress = True

    def m_init(self, **kwargs):
        xp = pxu.get_array_module(self.y)
        mst = self._mstate

        # Initialize the position and amplitude of the diracs
        # mst["x"] = xp.array([], dtype=np.float64)
        # mst["a"] = xp.array([], dtype=np.float64)
        mst["x"] = np.array([])
        mst["a"] = np.array([])

        mst["correction_prec"] = self._init_correction_prec
        mst["correction_iterations"] = []
        mst["correction_durations"] = []

    def swarm_init(self):
        mst = self._mstate
        # Initialize the particles and velocities
        particles = np.random.uniform(low=self.swarm_bounds[0], high=self.swarm_bounds[1], size=(self.swarm_n_particles, self.x_dim))
        mst['particles'] = particles
        mst['velocities'] = np.zeros((self.swarm_n_particles, self.x_dim))

        # Initialize the best positions and best costs
        mst['best_positions'] = particles.copy()
        mst['best_costs'] = self.dual_certificate(mst["particles"])

        # Initialize the global best position and global best cost
        mst['global_best_position'] = particles[0].copy()
        mst['global_best_cost'] = mst['best_costs'][0]

    def swarm_step(self):
        """
        Perform a particle swarm optimization step to find a list of dirac positions candidates.
        Code from: https://github.com/bnsreenu/python_for_microscopists/blob/master/321_what_is_particle_swarm_optimization.ipynb
        """
        mst = self._mstate

        self.swarm_init()

        for i in range(self.swarm_iterations):
            # Random matrix used to compute the cognitive component of the velocity update
            r1 = np.random.rand(self.swarm_n_particles, self.x_dim)
            # Random matrix used to compute the social component of the velocity update
            r2 = np.random.rand(self.swarm_n_particles, self.x_dim)

            # Cognitive component is calculated by taking the difference between the
            # particle's current position and its best personal position found so far,
            # and then multiplying it by a random matrix r1 and a cognitive acceleration coefficient c1.
            cognitive_component = self.swarm_c1 * r1 * (mst['best_positions'] - mst['particles'])

            # The social component represents the particle's tendency to move towards the
            # global best position found by the swarm. It is calculated by taking the
            # difference between the particle's current position and the global best position
            # found by the swarm, and then multiplying it by a random matrix r2 and a
            # social acceleration coefficient c2.
            social_component = self.swarm_c2 * r2 * (mst['global_best_position'] - mst['particles'])

            # The new velocity of the particle is computed by adding the current velocity
            # to the sum of the cognitive and social components, multiplied by the inertia
            # weight w. The new velocity is then used to update the position of the
            # particle in the search space.
            mst['velocities'] = self.swarm_w * mst['velocities'] + cognitive_component + social_component

            # Update the particles
            mst['particles'] += mst['velocities']

            # Enforce the bounds of the search space
            mst['particles'] = np.clip(mst['particles'], self.swarm_bounds[0], self.swarm_bounds[1])

            # Evaluate the objective function
            costs = self.dual_certificate(mst["particles"])

            # Update the best positions and best costs
            is_best = costs > mst['best_costs']
            mst['best_positions'][is_best] = mst['particles'][is_best]
            mst['best_costs'][is_best] = costs[is_best]

            # Update the global best position and global best cost
            global_best_index = np.argmax(mst['best_costs'])
            mst['global_best_position'] = mst['best_positions'][global_best_index].copy()
            mst['global_best_cost'] = mst['best_costs'][global_best_index]

            # if self.verbose:
            #     print(f"Iteration {i + 1}: Best Cost = {mst['global_best_cost'].item()}")

        if self.verbose:
            print(f"Global Best Position: {mst['global_best_position'].ravel()}")

    def correction_step(self) -> pxt.NDArray:
        r"""
        Method to update the weights after the selection of the new atoms. It solves a LASSO problem with a
        restricted support corresponding to the current set of active indices (active atoms). As mentioned,
        this method should be overriden in a child class for case-specific improved implementation.

        Returns
        -------
        weights: NDArray
            New iterate with the updated weights.
        """
        mst = self._mstate

        def correction_stop_crit(eps) -> StoppingCriterion:
            stop_crit = pxos.RelError(
                eps=eps,
                var="x",
                f=self.dual_certificate,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        print(mst["x_candidates"].shape)
        print(mst["a_candidates"].shape)
        data_fid = self.data_fid(mst["x_candidates"])

        a0 = mst["a_candidates"]
        dim = a0.shape[0]

        penalty = pxop.L1Norm(dim)
        apgd = PGD(data_fid, self.lambda_ * penalty, show_progress=self.show_progress)

        min_iter = pxos.MaxIter(n=self._min_correction_steps)
        max_duration = pxos.MaxDuration(t=dt.timedelta(seconds=15))

        stop = (min_iter & correction_stop_crit(self._mstate["correction_prec"])) | max_duration | pxos.MaxIter(100)

        apgd.fit(
            x0=a0,
            tau=1 / data_fid.diff_lipschitz,
            stop_crit=stop,
        )
        # mst["correction_iterations"].append(apgd.stats()[1]["iteration"][-1])
        # mst["correction_durations"].append(apgd.stats()[1]["duration"][-1])
        sol, _ = apgd.stats()

        updated_a = sol["x"]
        print(f"Updated a: {updated_a}")
        return updated_a

    def data_fid(self, support_indices: pxt.NDArray) -> pxo.DiffFunc:
        forward_op = self.tmp(support_indices)
        data_fid = 0.5 * pxop.SquaredL2Norm(dim_shape=forward_op.codim_shape[0]).argshift(-self.y) * forward_op
        # t1 = time()
        # data_fid.estimate_diff_lipschitz()
        # print("Time to estimate diff lipschitz: ", time() - t1)
        data_fid.diff_lipschitz = 1
        return data_fid

    def get_forward_operator(self, x: pxt.NDArray) -> MyLinOp:
        self.forward_op = self.forward_op.get_new_operator(x)
        return self.forward_op

    def tmp(self, x: pxt.NDArray) -> MyLinOp:
        return TMP(x, self.forward_op.n_measurements, self.forward_op.fc)

    def m_step(self):
        # Find a list of dirac positions candidates
        self.swarm_step()

        mst = self._mstate
        xp = pxu.get_array_module(self.y)

        if self.verbose:
            print("Before correction step:")
            print(f"Number of atoms: {len(mst['x'])}")
            print(f"Positions: {mst['x'].ravel()}")
            print(f"Amplitudes: {mst['a'].ravel()}")

        # mst["x_candidates"] = xp.concatenate([mst["x"].reshape(-1, 1), mst["best_positions"]])
        # mst["a_candidates"] = xp.concatenate([mst["a"], mst["best_costs"] / xp.linalg.norm(mst["best_costs"])])

        # mst["correction_prec"] = max(self._init_correction_prec / self._astate["idx"], self._final_correction_prec)
        # updated_a = self.correction_step()
        #
        # indices = updated_a > self.amplitude_threshold

        print(f"Loss: {self.dual_certificate(mst['global_best_position'])}")
        # mst["x"] = mst["x_candidates"][indices]
        # mst["a"] = updated_a[indices]

        mst["x"] = xp.append(mst["x"], mst["global_best_position"])
        mst["a"] = xp.append(mst["a"], 1)

        if self.verbose:
            print("After correction step:")
            print(f"Number of atoms: {len(mst['x'])}")
            print(f"Positions: {mst['x'].ravel()}")
            print(f"Amplitudes: {mst['a'].ravel()}")

    def solution(self) -> pxt.NDArray:
        return self._mstate['x'], self._mstate['a']

    def default_stop_crit(self) -> StoppingCriterion:
        # stop_crit = pxos.RelError(
        #     eps=1e-4,
        #     var="x",
        #     f=None,
        #     norm=2,
        #     satisfy_all=True,
        # )
        # return stop_crit
        return pxos.MaxIter(7)

    def objective_func(self) -> pxt.NDArray:
        return self.dual_certificate(self._mstate['x'])

    def dual_certificate(self, x: pxt.NDArray) -> pxt.NDArray:
        mst = self._mstate
        dual_cert = DualCertificate(mst["x"], mst["a"], self.y, self.get_forward_operator(mst["x"]), self.lambda_)
        return dual_cert.apply(x)
