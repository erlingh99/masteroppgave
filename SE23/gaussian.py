from dataclasses import dataclass
import numpy as np
import scipy.stats
from shapely import geometry as sg, ops as so
from .lie_theory import LieGroup


@dataclass
class MultiVarGauss:
    """A class for using Gaussians"""
    mean: np.ndarray
    cov: np.ndarray

    def __post_init__(self):
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matricies must be symmetric")

    @property
    def ndim(self) -> int:
        return self.mean.shape[0]

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Calculate the mahalanobis distance between self and x.
        """
        err = x.reshape(-1, 1) - self.mean.reshape(-1, 1)
        mahalanobis_distance = float(err.T @ np.linalg.solve(self.cov, err))
        return mahalanobis_distance

    def mahal_dist(self, x: np.ndarray) -> float:
        return self.mahalanobis_distance(x)

    def pdf(self, x: np.ndarray) -> float:
        """Calculate the likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).pdf(x)

    def logpdf(self, x: np.ndarray) -> float:
        """Calculate the log likelihood of x given the Gaussian"""
        return scipy.stats.multivariate_normal(self.mean, self.cov).logpdf(x)

    def sample(self) -> np.ndarray:
        """Sample from the Gaussian"""
        noise = np.random.multivariate_normal(
            np.zeros_like(self.mean), self.cov, 1).reshape(-1)
        return self.mean + noise

    def get_marginalized(self, indices) -> "MultiVarGauss":
        i_idx, j_idx = np.meshgrid(indices, indices,
                                   sparse=True, indexing='ij')
        mean = self.mean[i_idx.ravel()]
        cov = self.cov[i_idx, j_idx]
        return MultiVarGauss(mean, cov)

    def cholesky(self) -> np.ndarray:
        return np.linalg.cholesky(self.cov)
    
    @staticmethod
    def __hyperspherical_coords__(ndim, resolution):
        u = np.linspace(0, np.pi, resolution+1)
        v = np.linspace(0, 2*np.pi, resolution+1)

        spherical_coords = np.meshgrid(*np.array([u]*(ndim-2)), v)
        # need only ndim - 1 spherical coords (r is constant on the hyper-sphere)
        cart_coord = np.empty((ndim, (resolution+1)**(ndim-1)))
        # create cartesian coords
        for d in range(ndim):
            x_ns = 1
            for dd in range(d+1):
                if dd == ndim-1:
                    break
                if dd == d:
                    x_ns *= np.cos(spherical_coords[dd])
                else:
                    x_ns *= np.sin(spherical_coords[dd])

            cart_coord[d, :] = x_ns.T.ravel()
        return cart_coord
        
    def covar_ellipsis(self, num_std=3, resolution=20) -> np.ndarray:        
        return self.__covar_helper__(num_std, resolution) + self.mean[:, np.newaxis]


    def __covar_helper__(self, num_std=3, resolution=20) -> np.ndarray:    
        eig_vals, lambda_, _ = np.linalg.svd(self.cov)
        lambda_root = np.sqrt(lambda_)
        
        idx = np.argsort(-lambda_root)
        lambda_root = lambda_root[idx]
        eig_vals = eig_vals[:, idx]

        ndim = self.cov.shape[0] #the number of dimensions, aka the number of cartesian coordinates
        cart_coord = MultiVarGauss.__hyperspherical_coords__(ndim, resolution)        
        return eig_vals@np.diag(lambda_root)@cart_coord*num_std


    def draw_significant_ellipses(self, ax, n_ellipsis=3, n_std=3, n_points=50, color="red"):
        """
        Draw ellipse based on the 3 more important directions of the covariance
        """
        n = min(self.mean.shape[0], 3)
        cov = self.cov[:n, :n]
        mean = self.mean[:n]

        n_ellipsis = max(1, min(n_ellipsis, len(self.mean)-1, 3))

        idxs = [[0, 1], [0, 2], [1, 2]]


        V, eig_vals, _ = np.linalg.svd(cov)
        eig_root = np.sqrt(eig_vals)
        dirs = n_std*V@np.diag(eig_root)
        idx = np.argsort(-eig_vals) #find the index of the biggest eigenvalues, = the smallest when negating
       
        coords = MultiVarGauss.__hyperspherical_coords__(2, n_points)

        lines = np.empty((coords.shape[1], mean.shape[0]))

        for n in range(n_ellipsis):
            xi = dirs[:, idx[idxs[n]]]@coords

            for i, x in enumerate(xi.T):
                lines[i, :] = x + mean

            if mean.shape[0] > 2:
                ax.plot(*lines.T, color=color, alpha=0.4)                
            else:
                ax.fill(*lines[:, :2].T, color=color, alpha=0.2)


    
    def draw_significant_spheres(self, ax, n_spheres=4, n_std=3, n_points=50, color="rgbk"):
        """
        Draw spheres (3d) based on the 1 to 3 more important directions of the covariance
        """
        n_spheres = max(1, min(n_spheres, 4))

        idxs = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        V, eig_vals, _ = np.linalg.svd(self.cov)
        eig_root = np.sqrt(eig_vals)
        dirs = n_std*V@np.diag(eig_root)
        idx = np.argsort(-eig_vals) #find the index of the biggest eigenvalues, = the smallest when negating

        cart_coords = MultiVarGauss.__hyperspherical_coords__(ndim=3, resolution=n_points) #coordinates of identity sphere

        lines = np.empty((len(cart_coords[0]), 3))

        for n in range(n_spheres):
            
            xi = dirs[:, idx[idxs[n]]]@cart_coords

            for i, x in enumerate(xi.T):
                lines[i] = self.mean + x

            # ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], color=color[n])
            ax.plot_surface(lines[:, 0].reshape(n_points+1, -1), lines[:, 1].reshape(n_points+1, -1), lines[:, 2].reshape(n_points+1, -1), alpha=0.4, color=color[n])
    



    def __iter__(self):
        """
        Enable iteration over the mean and covariance. i.e.
            mvg = MultiVarGauss(mean=m, cov=c)
            mean, cov = mvg
        """
        return iter((self.mean, self.cov))

    def __repr__(self) -> str:
        """Used for pretty printing"""
        def sci(x): return f"{float(x):.23}"
        out = '\n'
        for i in range(self.mean.shape[0]):
            mline = sci(self.mean[i])
            cline = ' |'.join(sci(self.cov[i, j])
                              for j in range(self.cov.shape[1]))
            out += f"|{mline} |      |{cline} |\n"
        return out

    def __str__(self) -> str:
        return self.__repr__()
    
@dataclass
class ExponentialGaussian(MultiVarGauss):
    mean: LieGroup #many methods of MultiVarGauss is not well defined for this use case
    cov: np.ndarray

    def __post_init__(self):
        assert isinstance(self.mean, LieGroup), "The mean of an ExponentialGaussian must be a LieGroup" 
        assert self.cov.shape == (self.mean.ndim, self.mean.ndim), f"The covariance must be of shape {(self.mean.ndim, self.mean.ndim)}."

    def covar_ellipsis(self, num_std=3, resolution=20) -> np.ndarray:        
        xi = self.__covar_helper__(num_std, resolution)

        transformed = np.empty(xi.shape[1], dtype=object)
        for i in range(xi.shape[1]):       
            transformed[i] = self.mean@self.mean.__class__.Exp(xi[:, i])
        return transformed
      
    def project_translation(self, num_std=3, n_points=20): #project uncertainty ellipsis from exponential space. Also use to extract marginal covariances for plotting
        """
        Creates hyper-ellipsis in exp-space (ndim = LieGroup.ndim), 
        use exp-map of LieGroup to extract the values to plot from each pose, eg translation xyz
        """
        transformed = self.covar_ellipsis(num_std, n_points)
        return np.array([t.t for t in transformed])
    
    def draw_2Dtranslation_covariance_ellipse(self, ax, dir="xy", num_std=3, n_points=20, color="green"):
        if dir=="xy":
            I = [0, 1]
        elif dir == "xz":
            I = [0, 2]
        elif dir == "yz":
            I = [1, 2]
        else:
            raise ValueError("dir must be one of 'xy', 'xz', 'zy'.")

        def extract_polygon_slices(grid_2d):
            p_a = grid_2d[:-1, :-1]
            p_b = grid_2d[:-1, 1:]
            p_c = grid_2d[1:, 1:]
            p_d = grid_2d[1:, :-1]

            quads = np.concatenate((p_a, p_b, p_c, p_d), axis=2)

            m, n, _ = grid_2d.shape
            quads = quads.reshape(((m-1) * (n-1), 4, 2))

            return [sg.Polygon(t).buffer(0.0001, cap_style=2, join_style=2) for t in quads]

   
        coords = self.project_translation(num_std, n_points)

        p_grid = np.reshape(coords[:, I], [-1, (n_points + 1), 2])
        polygons = extract_polygon_slices(p_grid)
        union = so.unary_union(polygons)
        if not union.geom_type == 'Polygon':
            print("Error generating covariance polygon, plotting the raw points...")
            ax.plot(coords[:, I[0]], coords[:, I[1]])
            return
        
        ax.fill(*union.exterior.xy, alpha=0.1, facecolor=color)
        ax.plot(*union.exterior.xy, color=color)  

    def draw_significant_ellipses(self, ax, n_ellipsis=3, n_std=3, n_points=50, color="red"):
        """
        Draw ellipse based on the 3 more important directions of the covariance
        """
        n_ellipsis = max(1, min(n_ellipsis, 3))

        idxs = [[0, 1], [0, 2], [1, 2]]

        # eig_vals, V = np.linalg.eig(self.cov)
        V, eig_vals, _ = np.linalg.svd(self.cov)
        

        eig_root = np.sqrt(eig_vals)
        dirs = n_std*V@np.diag(eig_root)
        idx = np.argsort(-eig_vals) #find the index of the biggest eigenvalues, = the smallest when negating
       
        coords = MultiVarGauss.__hyperspherical_coords__(2, n_points)


        ndim = 3 if ax.name == "3d" else 2
        lines = np.empty((coords.shape[1], ndim))

        for n in range(n_ellipsis):
            xi = dirs[:, idx[idxs[n]]]@coords

            for i, x in enumerate(xi.T):
                Ttemp = self.mean@self.mean.__class__.Exp(x)
                lines[i] = Ttemp.t[:ndim]

            if ndim == 3:
                ax.plot(*lines.T, color=color, alpha=0.4)
            else:
                ax.fill(*lines.T, color=color, alpha=0.2)


    def draw_significant_spheres(self, ax, n_spheres=4, n_std=3, n_points=50, color="rgbk"):
        """
        Draw spheres (3d) based on the 1 to 3 more important directions of the covariance
        """
        n_spheres = max(1, min(n_spheres, 4))

        idxs = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

        V, eig_vals, _ = np.linalg.svd(self.cov)
        eig_root = np.sqrt(eig_vals)
        dirs = n_std*V@np.diag(eig_root)
        idx = np.argsort(-eig_vals) #find the index of the biggest eigenvalues, = the smallest when negating

        cart_coords = MultiVarGauss.__hyperspherical_coords__(ndim=3, resolution=n_points) #coordinates of identity sphere

        lines = np.empty((len(cart_coords[0]), 3))

        for n in range(n_spheres):
            
            xi = dirs[:, idx[idxs[n]]]@cart_coords

            for i, x in enumerate(xi.T):
                Ttemp = self.mean@self.mean.__class__.Exp(x)
                lines[i] = Ttemp.t

            # ax.plot(lines[:, 0], lines[:, 1], lines[:, 2], color=color[n])
            ax.plot_surface(lines[:, 0].reshape(n_points+1, -1), lines[:, 1].reshape(n_points+1, -1), lines[:, 2].reshape(n_points+1, -1), alpha=0.4, color=color[n])
    
    
    def copy(self):
        return ExponentialGaussian(self.mean.copy(), self.cov.copy())
    
    def __str__(self) -> str:
        mean_string = self.mean.__str__()
        cov_string = np.array2string(self.cov, precision=2, max_line_width=100)
        return "ExponentialGaussian\nMean:\n" + mean_string + "\nCovariance:\n" + cov_string