
####################### Question 1 ######################

# transition matrix P
d1 <- 0.65
d2 <- 0.40
c1 <- 1 - d1
c2 <- 1 - d2
P <- matrix(c(d1, c1, 0, 0,
              d1, 0, c1, 0,
              0, c2, 0, d2,
              0, 0, c2, d2), 
            nrow = 4, byrow = TRUE)

# initial state vector
p0 <- c(1/4, 1/4, 1/4, 1/4)

# one-step transition matrix P^2
P2 <- P %*% P

# state at time k+1 for k >= 1
kmax <- 100
pk <- p0
Evals <- numeric(kmax+1)
Evals[1] <- (pk[1] + pk[2]) * d1 + (pk[3] + pk[4]) * d2
for (i in 1:kmax) {
  pk <- P %*% pk
  Evals[i+1] <- (pk[1] + pk[2]) * d1 + (pk[3] + pk[4]) * d2
}
print(pk)

# Plot E(k) vs k
plot(0:kmax, Evals, type = "l", xlab = "k", ylab = "Expected number of visits to state 1")


# Computing stationary distribution pi
e <- eigen(t(P))
pi <- e$vectors[, 1] / sum(e$vectors[, 1])
print(pi)

# Check if P* = P^t * P
left <- pi %*% P
right <- pi
print(all.equal(left, right))

################### Question 2 ##################################3


# transition matrix P
d1 <- 0.64
d2 <- 0.40
c1 <- 1 - d1
c2 <- 1 - d2
P <- matrix(c(d1, c1, 0, 0, 0, 0,
              d1, 0, c1, 0, 0, 0,
              d1, 0, 0, c1, 0, 0,
              0, 0, c2, 0, 0, d2,
              0, 0, 0, c2, 0, d2,
              0, 0, 0, 0, c2, 0), nrow = 6, byrow = TRUE)

# One-step transition matrix P^2
P2 <- P %*% P
print(P2)

# initial state p(0)
p0 <- c(1/6, 1/6, 1/6, 1/6, 1/6, 1/6)

# state at time k+1 for k >= 1
kmax <- 100
pk <- p0
Evals <- numeric(kmax+1)
Evals[1] <- (pk[1] + pk[2] + pk[3]) * d1 + (pk[4] + pk[5] + pk[6]) * d2
for (i in 1:kmax) {
  pk <- P %*% pk
  Evals[i+1] <- (pk[1] + pk[2] + pk[3]) * d1 + (pk[4] + pk[5] + pk[6]) * d2
}
print(pk)

# Plot E(k) vs k
plot(0:kmax, Evals, type = "l", xlab = "k", ylab = "Expected number of visits to state 1")

# Computing stationary distribution pi
e <- eigen(t(P))
pi <- e$vectors[, 1] / sum(e$vectors[, 1])
print(pi)

# Check if P* = P^t * P
left <- pi %*% P
right <- pi
print(all.equal(left, right))


