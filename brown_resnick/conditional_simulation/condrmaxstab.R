library(SpatialExtremes)


x <- seq(-10, 10, length.out = 3)
y <- seq(-10, 10, length.out = 3)
grid_coords <- as.matrix(expand.grid(x = x, y = y))
set.seed(123)
range_param <- 1  # range parameter
smooth_param <- 1  # smoothness parameter (for Whittle-Matern covariance)
nugget_param <- 0  # nugget effect (set to 0 for no nugget)
data <- rmaxstab(1, grid_coords, "powexp", range = range_param, smooth = smooth_param, nugget = nugget_param)
observed_indices <- sample(1:nrow(grid_coords), 2)
missing_indices <- setdiff(1:nrow(grid_coords), observed_indices)

observed_coords <- grid_coords[observed_indices, ]
observed_data <- data[, observed_indices]





missing_coords <- grid_coords[missing_indices, ]

set.seed(123)
cond_sample <- condrmaxstab(
  k = 1,
  coord = as.matrix(observed_coords),
  cond.data = observed_data,
  cond.coord = as.matrix(missing_coords),
  cov.model = "brown",
  range = range_param,
  smooth = smooth_param,
  nugget = nugget_param
)