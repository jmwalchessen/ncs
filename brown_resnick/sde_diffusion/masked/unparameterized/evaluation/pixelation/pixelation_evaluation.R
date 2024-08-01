
library(reticulate)
library(devtools)
library(sp)
working_directory <- (strsplit(getwd(), "/sde_diffusion")[[1]])[1]
spatialextremes_directory <- paste(working_directory, "my-spatial-extremes", sep = "/")
devtools::install(spatialextremes_directory)
library(gstat)

number_of_replicates <- 1
n.size <- 1024
nn <- sqrt(n.size)
x <- y <- seq(-10, 10, length = nn)
coord <- expand.grid(x, y)
names(coord) <- c("x", "y")
range <- 1.6
smooth <- 1.6
z <- SpatialExtremes::rmaxstab(n = number_of_replicates, coord = coord, cov.mod = "brown", range = range, smooth = smooth)
z <- as.vector(z[1,])
data_df <- data.frame(x = coord$x, y = coord$y, z = z)
sp::coordinates(data_df) <- ~x + y
emp_variogram <- gstat::variogram(z~1, data = data_df, cutoff = 1)
vfit <- gstat::fit.variogram(emp_variogram, model = vgm(range = 1.6, nugget = 0, psil = 10, model = "Exp"))
print(emp_variogram)
#plot(emp_variogram, model[2,])

n_simulations <- 10
simulations <- list()
for (i in 1:n_simulations) {
  sim_data <- SpatialExtremes::rmaxstab(n = 1, coord = coord, cov.mod = "brown", range = range,
  smooth = smooth)
  simulations[[i]] <- as.vector(sim_data[1,])
}
combined_data <- do.call(cbind, simulations)
data_list <- lapply(1:n_simulations, function(i) data.frame(x = coord$x, y = coord$y, z = combined_data[, i], sim = i))
combined_df <- do.call(rbind, data_list)
coordinates(combined_df) <- ~x + y
emp_variogram <- variogram(z ~ 1, data = combined_df, cutoff = 5)
