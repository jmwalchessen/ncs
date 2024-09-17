library(SpatialExtremes)

n.sim <- 5
n.cond <- 5
range <- 2
smooth <- 1.2
n.site <- 200
coord <- seq(-5, 5, length = n.site)


cond.coord <- seq(-4, 4, length = n.cond)
all.coord <- c(cond.coord, coord)
print("a")
all.cond.data <- rmaxstab(1, all.coord, "brown", nugget = 0.001, range = range,
smooth = smooth)
print("b")
cond.data <- all.cond.data[1:n.cond]
ans <- condrmaxstab(n.sim, coord, cond.coord, cond.data, range = range,
smooth = smooth, nugget = .001, cov.mod = "brown")

