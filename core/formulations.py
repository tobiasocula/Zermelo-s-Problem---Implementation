from core.integrating import *
from core.iterations_and_other import *

class Formulation:
    def __init__(self, A, B, k, funcs):
        self.A = A
        self.B = B
        self.k = k
        self.funcs = funcs

        self.results_shooting = None
        self.results_wind_adj = None
        self.results_branch_wind_adj = None

    def solve_shooting(self, phi_list, T_list, num_iters, num_points):
        self.results_shooting = iterate(phi_list, T_list, self.funcs, self.A, self.B, self.k, num_points, num_iters)
        return self

    def solve_wind_adj(self, stepsize, num_points, num_iters):
        self.results_wind_adj = iterative_wind(
            self.funcs, A=self.A, B=self.B, k=self.k, numpoints=num_points, num=num_iters, stepsize=stepsize
            )
        return self

    def solve_branch_wind_adj(self, stepsize, num_points, num_iters, dphi, num_phi_branches):
        self.results_branch_wind_adj = branch_iterative_wind(
            self.funcs, self.A, self.B, self.k, num_points, num_iters, num_phi_branches, dphi, stepsize
        )
        return self

    def plot_shooting(self):
        assert self.results_shooting is not None
        best_x, best_y = self.results_shooting[:2]
        fig, ax = plt.subplots()
        ax.plot(best_x, best_y)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

    def plot_wind_adj(self):
        assert self.results_wind_adj is not None

        x = self.results_wind_adj[0]
        y = self.results_wind_adj[1]

        fig, ax = plt.subplots()
        for xset, yset in zip(x,y):
            ax.plot(xset, yset)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

    def plot_branch_wind_adj(self):
        assert self.results_branch_wind_adj is not None
        x, y, _, _ = self.results_branch_wind_adj
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.scatter(self.A, self.B, color="red")
        ax.scatter(0, 0, color="green")
        plt.show()

def plot_formulations_together(forms, method):
    """method = shooting, wind_adj, branch_wind_adj"""
    if method == "shooting":
        assert all([f.results_shooting is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            best_x, best_y = f.results_shooting[:2]
            ax.plot(best_x, best_y)
            ax.scatter(f.A, f.B, color="red")
    elif method == "wind_adj":
        assert all([f.results_wind_adj is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            x, y, _ = f.results_wind_adj
            ax.plot(x[-1], y[-1])
            ax.scatter(f.A, f.B, color="red")
    elif method == "branch_wind_adj":
        assert all([f.results_branch_wind_adj is not None] for f in forms)
        fig, ax = plt.subplots()
        ax.scatter(0, 0, color="green")
        for f in forms:
            x, y, _, _ = f.results_branch_wind_adj
            ax.plot(x, y)
            ax.scatter(f.A, f.B, color="red")
    else:
        raise

    plt.show()


    
