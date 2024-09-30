import datetime as dt
from collections import abc as cabc
from time import time
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pyxu.abc.operator as pxo
import pyxu.abc.solver as pxs
import pyxu.info.ptype as pxt
import pyxu.operator as pxop
import pyxu.opt.stop as pxos
from pyxu.abc import StoppingCriterion
from pyxu.opt.solver.pgd import PGD
from pyxu.util import as_real_op, view_as_real, view_as_complex
from scipy.optimize import linprog, minimize

from src.operators.fourier_operator import DualCertificate, DiffFourierOperator
from src.operators.my_lin_op import MyLinOp


class FW(pxs.Solver):

    def __init__(
            self,
            measurements: pxt.NDArray,
            forward_op: MyLinOp,
            lambda_: float,
            x_dim: int,
            bounds: pxt.NDArray,
            verbose: bool = False,
            options: Dict[str, Any] = None,
            **kwargs):
        super().__init__(**kwargs)

        self.y = measurements
        self.forward_op = forward_op
        self.lambda_ = lambda_
        self.x_dim = x_dim
        self.bounds = bounds

        if options is None:
            options = {}

        # Initialize the swarm parameters
        self.swarm_iterations = options.get("swarm_iterations", 10)
        self.swarm_n_particles = options.get("swarm_n_particles", 100)
        self.swarm_w = options.get("swarm_w", 0.25)
        self.swarm_c1 = options.get("swarm_c1", 0.75)
        self.swarm_c2 = options.get("swarm_c2", 0.25)

        self.amplitude_threshold = options.get("amplitude_threshold", 0.)
        self.merge_threshold = options.get("merge_threshold", 0.01)

        self.verbose = verbose

        self.merge = options.get("merge", False)
        self.add_one = options.get("add_one", False)
        self.sliding = options.get("sliding", False)
        self.simplex = options.get("simplex", False)

        self.max_iter = options.get("max_iter", 100)

        self.min_iter = options.get("min_iter", 2)
        self.dual_certificate_tol = options.get("dual_certificate_tol", 1e-2)

    def m_init(self, **kwargs):
        mst = self._mstate

        # Initialize the position and amplitude of the diracs
        mst["x"] = np.array([], dtype=np.float64)
        mst["a"] = np.array([], dtype=np.float64)

        mst["swarm_durations"] = []

        mst["correction_iterations"] = []
        mst["correction_durations"] = []

        mst["sliding_durations"] = []

        # Plot
        mst["iter_candidates"] = []
        mst["iter_x"] = []
        mst["iter_a"] = []

    def swarm_init(self):
        mst = self._mstate
        # Initialize the particles and velocities
        particles = np.random.uniform(low=self.bounds[0], high=self.bounds[1],
                                      size=(self.swarm_n_particles, self.x_dim))
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
            mst['particles'] = np.clip(mst['particles'], self.bounds[0], self.bounds[1])

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

        def correction_stop_crit(eps) -> pxs.StoppingCriterion:
            stop_crit = pxos.RelError(
                eps=eps,
                var="x",
                f=None,
                norm=2,
                satisfy_all=True,
            )
            return stop_crit

        data_fid = self.data_fid(mst["x_candidates"])

        a0 = mst["a_candidates"]
        dim = a0.shape[0]

        if self._astate["idx"] == 1:
            a0[0] = 1e-5

        penalty = pxop.L1Norm(dim)
        apgd = PGD(data_fid, self.lambda_ * penalty, show_progress=False)

        min_iter = pxos.MaxIter(n=10)
        max_duration = pxos.MaxDuration(t=dt.timedelta(seconds=60))

        stop = (min_iter & correction_stop_crit(1e-4)) | max_duration

        apgd.fit(
            x0=a0,
            tau=1 / data_fid.diff_lipschitz,
            stop_crit=stop,
        )
        mst["correction_iterations"].append(apgd.stats()[1]["iteration"][-1])
        sol, _ = apgd.stats()

        updated_a = sol["x"]

        mst["updated_a"] = updated_a

        indices = np.abs(updated_a) > self.amplitude_threshold

        x = mst["x_candidates"][indices]
        a = updated_a[indices]
        return x, a

    def data_fid(self, support_indices: pxt.NDArray) -> pxo.DiffFunc:
        """Get the data fidelity term"""
        forward_op = self.get_forward_operator(support_indices)
        y = self.y
        data_fid = 0.5 * pxop.SquaredL2Norm(dim_shape=y.shape).argshift(-y) * forward_op
        data_fid.diff_lipschitz = max(len(y) * len(forward_op.x) / 4, 2 * len(y))
        return data_fid

    def get_forward_operator(self, x: pxt.NDArray) -> MyLinOp:
        """Get new updated forward operator with the new positions of the atoms"""
        self.forward_op = self.forward_op.get_new_operator(x)
        return self.forward_op

    def m_step(self):
        # Find a list of dirac positions candidates
        t1 = time()
        self.swarm_step()
        t2 = time()

        mst = self._mstate
        mst["swarm_durations"].append(t2 - t1)

        if self.verbose:
            print(f"Swarm step duration: {t2 - t1}")
            print("Before correction step:")
            print(f"Number of atoms: {len(mst['x'])}")
            # print(f"Positions: {mst['x'].ravel()}")
            # print(f"Amplitudes: {mst['a'].ravel()}")

        mst["iter_candidates"].append(mst["best_positions"].copy())

        # Add new atoms to the current set of atoms
        if self.add_one:
            # Add the global best particles to the list of candidates
            mst["x_candidates"] = np.append(mst["x"], mst["global_best_position"])
            mst["a_candidates"] = np.append(mst["a"], 0)
        else:
            # Add all particles to the list of candidates
            mst["x_candidates"] = np.concatenate([mst["x"].reshape(-1, 1), mst["best_positions"]])
            mst["a_candidates"] = np.concatenate([mst["a"], np.zeros_like(mst["best_costs"])])

        t1 = time()
        # Correction step to update amplitudes and remove candidates with 0 amplitudes
        mst["x"], mst["a"] = self.correction_step()
        t2 = time()
        mst["correction_durations"].append(t2 - t1)

        mst["iter_x"].append(mst["x"].copy())
        mst["iter_a"].append(mst["a"].copy())

        if self.verbose:
            print(f"Correction step duration: {t2 - t1}")
            print("After correction step:")
            print(f"Number of atoms: {len(mst['x'])}")
            # print(f"Positions: {mst['x'].ravel()}")
            # print(f"Amplitudes: {mst['a'].ravel()}")

        if self.merge:
            if self.verbose:
                print("Before merge step:")
                print(f"Number of atoms: {len(mst['x'])}")
                # print(f"Positions: {mst['x'].ravel()}")
                # print(f"Amplitudes: {mst['a'].ravel()}")

            mst["x"], mst["a"] = self.merge_atoms(mst["x"], mst["a"])

            mst["iter_x"].append(mst["x"].copy())
            mst["iter_a"].append(mst["a"].copy())

        if self.simplex:
            mst["x"], mst["a"] = self.simplex_compute()

            mst["iter_x"].append(mst["x"].copy())
            mst["iter_a"].append(mst["a"].copy())

        if self.sliding:
            if self.verbose:
                print("Before sliding step:")
                print(f"Number of atoms: {len(mst['x'])}")
                # print(f"Positions: {mst['x'].ravel()}")
                # print(f"Amplitudes: {mst['a'].ravel()}")

            t1 = time()
            mst["x"], mst["a"] = self.sliding_compute()
            t2 = time()
            mst["sliding_durations"].append(t2 - t1)

            mst["iter_x"].append(mst["x"].copy())
            mst["iter_a"].append(mst["a"].copy())

            if self.verbose:
                print(f"Sliding step duration: {t2 - t1}")
                print("After sliding step:")
                print(f"Number of atoms: {len(mst['x'])}")
                # print(f"Positions: {mst['x'].ravel()}")
                # print(f"Amplitudes: {mst['a'].ravel()}")

    def solution(self) -> pxt.NDArray:
        return self._mstate['x'].ravel(), self._mstate['a'].ravel()

    def merged_solution(self) -> pxt.NDArray:
        return self.merge_atoms(self._mstate['x'], self._mstate['a'])

    def merge_atoms(self, x: pxt.NDArray, a: pxt.NDArray) -> pxt.NDArray:
        """
        Merges positions that are within a given threshold distance, with the merged
        position being the average of the original positions.
        """
        position_amplitude_list = list(zip(x, a))
        position_amplitude_list = [(x.item(), a) for x, a in position_amplitude_list]

        # Sort the list by position
        sorted_list = sorted(position_amplitude_list, key=lambda x: x[0])

        merged_list = []
        current_positions = []
        current_amplitudes = []

        for position, amplitude in sorted_list:
            if not current_positions:
                # Start with the first position
                current_positions.append(position)
                current_amplitudes.append(amplitude)
            else:
                if abs(position - current_positions[-1]) <= self.merge_threshold:
                    # Append to current group
                    current_positions.append(position)
                    current_amplitudes.append(amplitude)
                else:
                    # Compute the average of current positions and amplitudes
                    average_position = np.average(current_positions, weights=current_amplitudes)
                    sum_amplitude = sum(current_amplitudes)
                    merged_list.append((average_position, sum_amplitude))

                    # Start a new group
                    current_positions = [position]
                    current_amplitudes = [amplitude]

        # Don't forget to handle the last group
        if current_positions:
            average_position = sum(current_positions) / len(current_positions)
            amplitude = sum(current_amplitudes)
            merged_list.append((average_position, amplitude))

        return np.array([x for x, _ in merged_list]), np.array([a for _, a in merged_list])

    def simplex_compute(self):
        updated_a = self._mstate["updated_a"]

        s = self.forward_op.fourier.shape
        A = as_real_op(self.forward_op.fourier).reshape((s[0] * 2, s[1] * 2))
        a = view_as_real(updated_a.astype(np.complex128)).ravel()
        b = A @ a
        lp = linprog(c=np.ones_like(a), A_eq=A, b_eq=b, method='simplex',
                     options={"presolve": False, "tol": 1e-2})
        print(lp.success)
        print(lp.message)
        a = lp.x
        a = np.real(view_as_complex(a.reshape((a.shape[0] // 2, 2))).ravel())
        updated_a = a
        indices = np.abs(updated_a) > self.amplitude_threshold
        x = self._mstate["x_candidates"][indices]
        a = updated_a[indices]
        return x, a

    def sliding_compute(self):
        x_tmp = self._mstate["x"].copy().ravel()
        a_tmp = self._mstate["a"].copy().ravel()
        op = DiffFourierOperator.get_DiffFourierOperator(self.forward_op)

        def fun(xa):
            x, a = np.split(xa, 2)
            z = op(xa) - view_as_complex(self.y)
            return np.real(z.T.conj() @ z) / 2 + self.lambda_ * np.sum(np.abs(a))

        def grad(xa):
            x, a = np.split(xa, 2)
            z = op(xa) - view_as_complex(self.y)
            grad_x = a * (op.grad_x(xa) @ z)
            grad_a = op.grad_a(xa) @ z + self.lambda_ * np.sign(a)
            return np.real(np.concatenate([grad_x, grad_a]))

        op = minimize(fun, np.concatenate([x_tmp, a_tmp]), method="BFGS", jac=grad)
        return np.split(op.x, 2)

    def default_stop_crit(self) -> StoppingCriterion:
        stop_crit = StopDualCertificate(self.y, self.forward_op, self.lambda_, self.bounds, self.dual_certificate_tol)
        return (stop_crit & pxos.MaxIter(self.min_iter)) | pxos.MaxIter(self.max_iter)

    def objective_func(self) -> pxt.NDArray:
        return self.dual_certificate(self._mstate["x"])

    def dual_certificate(self, t: pxt.NDArray) -> pxt.NDArray:
        mst = self._mstate
        dual_cert = DualCertificate(mst["x"], mst["a"], self.y, self.get_forward_operator(mst["x"]), self.lambda_)
        return dual_cert.apply(t)

    def plot(self, x, a):
        """ Plot the results of the solver and each steps of the algorithm."""
        mst = self._mstate

        n_iter = len(mst["iter_x"])
        grid = np.linspace(self.bounds[0], self.bounds[1], 2048)

        need_extra_plot = self.merge or self.sliding or self.simplex

        if need_extra_plot:
            n_iter = n_iter // 2
            fig, axs = plt.subplots(n_iter, 3, figsize=(20, 5 * n_iter))
        else:
            fig, axs = plt.subplots(n_iter, 2, figsize=(10, 5 * n_iter))

        # Initial dual certificate
        dual_cert = DualCertificate(np.array([]), np.array([]), self.y, self.get_forward_operator(np.array([])),
                                    self.lambda_)
        eta = dual_cert(grid)

        for i in range(n_iter):
            # Normalize the dual certificate
            idx = i * 2 if need_extra_plot else i

            s1 = axs[i, 0].stem(mst["iter_candidates"][i], np.ones_like(mst["iter_candidates"][i]),
                                linefmt='k.--', markerfmt='k.', basefmt=" ", label='Candidates')
            ax2 = axs[i, 0].twinx()
            ax2.plot(grid, eta, label='Dual Certificate', color='tab:blue')
            axs[i, 0].set_title(f"Candidates - Iteration {i + 1}")
            axs[i, 0].grid(True)

            dual_cert = DualCertificate(mst["iter_x"][idx], mst["iter_a"][idx], self.y,
                                        self.get_forward_operator(mst["iter_x"][idx]), self.lambda_)
            eta = dual_cert(grid)
            ax2 = axs[i, 1].twinx()
            l1, = ax2.plot(grid, eta, label='Dual Certificate', color='tab:blue')
            s2 = axs[i, 1].stem(x, a, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
            s3 = axs[i, 1].stem(mst["iter_x"][idx], mst["iter_a"][idx], linefmt='r.--', markerfmt='ro', basefmt=" ",
                                label='Reconstruction')
            axs[i, 1].set_title(f"Correction - Iteration {i + 1}")
            axs[i, 1].grid(True)
            axs[i, 1].set_zorder(1)
            axs[i, 1].set_frame_on(False)

            if need_extra_plot:
                idx += 1

                dual_cert = DualCertificate(mst["iter_x"][idx], mst["iter_a"][idx], self.y,
                                            self.get_forward_operator(mst["iter_x"][idx]), self.lambda_)
                eta = dual_cert(grid)
                ax2 = axs[i, 2].twinx()
                ax2.plot(grid, eta, label='Dual Certificate', color='tab:blue')
                axs[i, 2].stem(x, a, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
                axs[i, 2].stem(mst["iter_x"][idx], mst["iter_a"][idx], linefmt='r.--', markerfmt='ro', basefmt=" ",
                               label='Reconstruction')
                # axs[i, 2].stem(mst["iter_x"][idx-1], mst["iter_a"][idx-1], linefmt='g.--', markerfmt='go', basefmt=" ",
                #                label='R')
                if self.merge:
                    name = "Merge"
                elif self.sliding:
                    name = "Sliding"
                elif self.simplex:
                    name = "Simplex"
                else:
                    name = "None"

                axs[i, 2].set_title(name + f" - Iteration {i + 1}")
                axs[i, 2].grid(True)
                axs[i, 2].set_zorder(1)
                axs[i, 2].set_frame_on(False)

            # Update the dual certificate
            idx = i * 2 if need_extra_plot else i
            dual_cert = DualCertificate(mst["iter_x"][idx], mst["iter_a"][idx], self.y,
                                        self.get_forward_operator(mst["iter_x"][idx]), self.lambda_)
            eta = dual_cert(grid)

            # Legend
            if i == 0:
                handles = [l1, s1, s2, s3]
                labels = ['Dual Certificate', 'Candidates', 'Ground Truth', 'Reconstruction']
                # Place legend outside the plot area
                fig.legend(handles, labels, loc='upper center', ncol=2)

        plt.show()
        # plt.savefig("results.png")

    def plot_solution(self, x0, a0, merged=False):
        if merged:
            x, a = self.merged_solution()
        else:
            x, a = self.solution()

        plt.figure(figsize=(10, 5))
        plt.stem(x0, a0, linefmt='k.--', markerfmt='ko', basefmt=" ", label='Ground Truth')
        plt.stem(x, a, linefmt='r.--', markerfmt='ro', basefmt=" ", label='Reconstruction')
        plt.title("Ground Truth vs Reconstruction")
        plt.grid(True)
        plt.legend()
        plt.show()


class StopDualCertificate(StoppingCriterion):

    def __init__(self, y: pxt.NDArray, forward_operator: MyLinOp, lambda_: float, bound: pxt.NDArray,
                 dual_certificate_tol: float = 1e-2):
        self._val = -1.
        self.y = y
        self.forward_operator = forward_operator
        self.lambda_ = lambda_
        self.dual_certificate_tol = dual_certificate_tol
        self.grid = np.linspace(bound[0], bound[1], 2048)

    def stop(self, state: cabc.Mapping[str]) -> bool:
        x, a = state["x"], state["a"]

        if len(x) == 0:
            return False

        self.forward_operator = self.forward_operator.get_new_operator(x)
        dual_cert = DualCertificate(x, a, self.y, self.forward_operator, self.lambda_)

        self._val = np.max(np.abs(dual_cert.apply(self.grid))).item()
        return np.isclose(self._val, 1, atol=self.dual_certificate_tol).item()

    def info(self) -> cabc.Mapping[str, float]:
        return {"Dual Certificate max value": self._val}
