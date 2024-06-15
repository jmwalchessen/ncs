library(spdep)
library(spGARCH)
library(pracma)
library(fastmatrix)

construct_norm_matrix <- function(minX, maxX, minY, maxY, n) {

    x <- seq(minX, maxX, length.out = n)
    y <- seq(minY, maxY, length.out = n)
    locations <- meshgrid(x,y)
    xlocations <- as.vector(locations$X)
    ylocations <- as.vector(locations$Y)
    X_matrix <- rep(locations$X, n**2)
    Y_matrix <- rep(locations$Y, n**2)
    dim(X_matrix) <- c(n**2,n**2)
    dim(Y_matrix) <- c(n**2,n**2)
    Xsquared = (X_matrix - t(X_matrix))*(X_matrix - t(X_matrix))
    Ysquared = (Y_matrix - t(Y_matrix))*(Y_matrix - t(Y_matrix))
    norm_matrix <- sqrt(Xsquared + Ysquared)
    return(norm_matrix)
    }

compute_bound <- function(rho, W)
{
    bound <- (matrix.norm(((rho**2)*(W %*% W)), type = "1"))**(-(1/4))
    return(bound)
}

construct_exp_kernel <- function(minX, maxX, minY, maxY, n, variance, lengthscale)
{
    norm_matrix = construct_norm_matrix(minX, maxX, minY, maxY, n)
    exp_kernel = variance*exp((-1/lengthscale)*norm_matrix)
    return(exp_kernel)
}

list_d_degree_queen_neighbors <- function(labelmatrix, d, i, j)
{
    current_neighbors <- c()
    current_label <- labelmatrix[i,j]
    if (((i >= d) & (j >= d)) & (i <= (n-d)) & (j <= (n-d)))
    {
        for(k in -d:d)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i < d) & (j >= d) & (j <= (n-d)))
    {
        for(k in -i:d)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i > n - d) & (j >= d) & (j <= (n-d)))
    {
        m = n-i
        for(k in -d:m)
        {
            for(l in -d:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i >= d) & (i <= (n-d)) & (j < d))
    {
       for(k in -d:d)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        } 
    }

    else if ((i >= d) & (i <= (n-d)) & (j > (n-d)))
    {
        for(k in -d:d)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i < d) & (j < d))
    {
        for(k in -i:d)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }

    else if ((i < d) & (j > (n-d)))
    {
        for(k in -i:d)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }

    else if ((i > (n-d)) & (j > (n-d)))
    {
        p = n-i
        for(k in -d:p)
        {
            m = n-j
            for(l in -d:m)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    else if ((i > (n-d)) & (j < d))
    {
        p = n-i
        for(k in -d:p)
        {
            for(l in -j:d)
            {
                if((k != 0) | (l != 0))
                {
                    current_neighbors <- c(current_neighbors, labelmatrix[i+k,j+l])
                }
            }
        }
    }
    return(current_neighbors)
}

construct_d_degree_queen_contiguity_list <- function(n,d)
{
    labelmatrix <- seq(1,n**2,1)
    dim(labelmatrix) <- c(n,n)
    neighbors_list <- list()
    for(i in 1:n)
    {
        for(j in 1:n)
        { 
            current_neighbors <- list_d_degree_queen_neighbors(labelmatrix, d, i, j)
            neighbors_list[[(j-1)*n+i]] <- as.integer(current_neighbors)
        }
    }
    return(neighbors_list)
}

n <- 32
d <- 1
labelmatrix <- seq(1,n**2,1)
dim(labelmatrix) <- c(n,n)
nba <- list_d_degree_queen_neighbors(labelmatrix, d, 1, 3)
neighbors_list <- construct_d_degree_queen_contiguity_list(n,d)
class(neighbors_list) <- c("nb")
nblist <- cell2nb(n, n, type = "queen")
#print(nblist[[1]])
#print(typeof(nblist[[1]]))
W <- nb2mat(neighbors_list)
rho <- 1
alpha <- 1.5
a <- compute_bound(rho, W)



y = sim.spARCH(n = dim(W)[1], rho = rho, alpha = alpha, W= W, type = "spARCH")