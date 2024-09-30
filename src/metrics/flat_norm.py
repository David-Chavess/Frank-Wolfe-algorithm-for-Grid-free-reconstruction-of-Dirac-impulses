import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

"""Code from https://github.com/qdenoyelle/OTMSL"""


# Structure to store output of flat_norm
class UOTOutput:
    def __init__(self, cost=float('inf'), P=None, C=None, mass_variation=None):
        self.cost = cost
        self.P = P if P is not None else np.zeros((1, 1))
        self.C = C if C is not None else np.zeros((1, 1))
        self.mass_variation = mass_variation if mass_variation is not None else np.zeros(1)


# Function to initialize UOT_output
def init_UOT_output(n=1, m=1):
    return UOTOutput(float('inf'), np.zeros((n, m)), np.zeros((n, m)), np.zeros(n + m))


# Euclidean distance function
def euclidean_distance(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))


# Function to compute flat metric
def flat_norm(x, y, a, b, lambda_val, coupling_mass_variation=False, ground_metric=euclidean_distance):
    n, m = len(x), len(y)
    out = init_UOT_output(n, m)
    c = np.concatenate([a, -b])

    # Define the optimization variable
    f = cp.Variable(n + m)

    # Create the list of constraints
    constraints = []
    for k in range(n):
        for l in range(m):
            e = ground_metric(x[k], y[l])
            constraints.append(-e <= f[k] - f[l + n])
            constraints.append(f[k] - f[l + n] <= e)

    # Box constraints: -lambda <= f <= lambda
    constraints.append(f >= -lambda_val)
    constraints.append(f <= lambda_val)

    # Define the objective function
    objective = cp.Maximize(f @ c)

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    # Store the optimal cost
    out.cost = prob.value

    # Compute the coupling and cost matrices if requested
    p = n * m
    if coupling_mass_variation:
        dual_values = np.zeros(p)
        for i in range(p):
            # Extract dual variables for the constraints
            lip_constr = constraints[2 * i + 1]  # one of the two constraints added for each pair (k, l)
            dual_value = lip_constr.dual_value
            dual_values[i] = dual_value

        out.mass_variation = np.array([constr.dual_value for constr in constraints[-2:]])  # dual of box constraints
        out.mass_variation = out.mass_variation.flatten()
        f_opt_value = f.value

        k = 0
        for i in range(n):
            for j in range(m):
                out.C[i, j] = f_opt_value[i] - f_opt_value[j + n]
                if abs(dual_values[k]) > 1e-8:
                    out.P[i, j] = dual_values[k]
                k += 1

    return out


def plot_distributions(x, y):
    x = np.array(x)
    y = np.array(y)

    plt.scatter(x[:, 0], x[:, 1], color='blue', label='Measure a (x)')
    plt.scatter(y[:, 0], y[:, 1], color='red', label='Measure b (y)', alpha=0.6)

    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('Point Source Distributions')
    plt.legend()
    plt.grid(True)
    plt.show()


def line(x1, x2):
    t = np.linspace(0, 1, 100)
    lx = [(1 - tt) * x1[0] + tt * x2[0] for tt in t]
    ly = [(1 - tt) * x1[1] + tt * x2[1] for tt in t]
    return lx, ly


def plot_points_sources(bounds, x, y, a, b):
    n, m = len(x), len(y)
    plt.figure(figsize=(5, 5))

    for k in range(n):
        ms = 10.0 * a[k] / (np.mean(a) + np.mean(b))
        if k < n - 1:
            plt.plot([x[k][0]], [x[k][1]], linestyle="none", marker=".", ms=ms, color="red")
        else:
            plt.plot([x[k][0]], [x[k][1]], linestyle="none", marker=".", ms=ms, color="red", label=r"$\mu_{GT}$")

    for k in range(m):
        ms = 10.0 * b[k] / (np.mean(a) + np.mean(b))
        if k < m - 1:
            plt.plot([y[k][0]], [y[k][1]], linestyle="none", marker=".", ms=ms, color="green")
        else:
            plt.plot([y[k][0]], [y[k][1]], linestyle="none", marker=".", ms=ms, color="green", label=r"$\mu_{est}$")

    ax = plt.gca()
    ax.set_xlim([bounds[0][0], bounds[1][0]])
    ax.set_ylim([bounds[0][1], bounds[1][1]])
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_points_sources_uot(x, y, a, b, bounds, uot_out):
    n, m = len(x), len(y)
    plt.figure(figsize=(5, 5))

    for k in range(n):
        ms = 25.0 * a[k] / (np.mean(a) + np.mean(b))
        if k < n - 1:
            plt.plot([x[k][0]], [x[k][1]], linestyle="none", marker=".", ms=ms, color="red")
        else:
            plt.plot([x[k][0]], [x[k][1]], linestyle="none", marker=".", ms=ms, color="red", label=r"$\mu_{GT}$")

    for k in range(m):
        ms = 25.0 * b[k] / (np.mean(a) + np.mean(b))
        if k < m - 1:
            plt.plot([y[k][0]], [y[k][1]], linestyle="none", marker=".", ms=ms, color="green")
        else:
            plt.plot([y[k][0]], [y[k][1]], linestyle="none", marker=".", ms=ms, color="green", label=r"$\mu_{est}$")

    # Plot transport plan lines based on uot_out.P
    for i in range(n):
        for j in range(m):
            if uot_out.P[i, j] > 1e-8:
                lx, ly = line(x[i], y[j])  # Assuming you have a corresponding function in Python
                lw = 2 * np.sum(uot_out.P > 0.0) * uot_out.P[i, j] * uot_out.C[i, j] / uot_out.cost
                if lw > 0.0:
                    plt.plot(lx, ly, linestyle="-", color="black")
                else:
                    print("### error ###")

    # Plot mass variation points
    for i in range(n):
        if abs(uot_out.mass_variation[i]) > 1e-8:
            ms = 25.0 * abs(uot_out.mass_variation[i]) / (np.mean(a) + np.mean(b))
            if uot_out.mass_variation[i] > 0.0:
                plt.plot([x[i][0]], [x[i][1]], linestyle="none", marker="+", ms=ms, color="black")
            else:
                plt.plot([x[i][0]], [x[i][1]], linestyle="none", marker="x", ms=ms, color="black")

    for j in range(m):
        if abs(uot_out.mass_variation[n + j]) > 1e-8:
            ms = 25.0 * abs(uot_out.mass_variation[n + j]) / (np.mean(a) + np.mean(b))
            if uot_out.mass_variation[n + j] > 0.0:
                plt.plot([y[j][0]], [y[j][1]], linestyle="none", marker="+", ms=ms, color="black")
            else:
                plt.plot([y[j][0]], [y[j][1]], linestyle="none", marker="x", ms=ms, color="black")

    ax = plt.gca()
    ax.set_xlim([bounds[0][0], bounds[1][0]])
    ax.set_ylim([bounds[0][1], bounds[1][1]])
    plt.title(r"Flat metric between $\mu_{est}$ to $\mu_{GT}$, cost=" + str(round(uot_out.cost, 5)))
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set the domain [0, 1] x [0, 1]
    bounds = [[0.0, 0.0], [1.0, 1.0]]

    x, y, a, b = np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), np.array([1.0]), np.array([0.5])
    output = flat_norm(x, y, a, b, 1.0)

    # Print results
    print(f"Cost: {output.cost}")
    print(f"Coupling matrix (P): \n{output.P}")
    print(f"Cost matrix (C): \n{output.C}")
    print(f"Mass variation: \n{output.mass_variation}")

    # exit(0)

    # Number of point sources
    K = 10

    # Support of first measure
    t_values = np.linspace(0, 1, K + 1)[:-1]
    x = [[0.5 + 0.25 * np.cos(2 * np.pi * t), 0.5 + 0.25 * np.sin(2 * np.pi * t)] for t in t_values]

    # Support of second measure with random noise
    y = [xi + 0.02 * np.random.randn(2) for xi in x]

    n, m = len(x), len(y)

    # Weights for the two measures
    a = np.ones(n) / n
    b = a + 0.2 * np.random.randn(n) / n  # Small random noise added to `b`

    # Regularization parameter (lambda)
    lambda_val = 0.02

    # Call the compute_flat_metric function
    output = flat_norm(x, y, a, b, lambda_val, coupling_mass_variation=True)

    # Print results
    print(f"Cost: {output.cost}")
    print(f"Coupling matrix (P): \n{output.P}")
    print(f"Cost matrix (C): \n{output.C}")
    print(f"Mass variation: \n{output.mass_variation}")

    # Optional: plot the point distributions
    # plot_distributions(x, y)

    # Call the plotting functions
    # plot_points_sources(bounds, x, y, a, b)
    plot_points_sources_uot(x, y, a, b, bounds, output)
