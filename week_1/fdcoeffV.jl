function fdcoeffV(k,x̄,x)
    n = length(x)
    if n <= k
        error("Too few points in the stencil")
    end
    A = zeros(n,n)
    A[1,:] = ones(n)
    for i = 2:n
        A[i,:] = @. (x - x̄)^(i-1) / factorial(i-1) # ∑ a_i * (x_i - x̄)^i/i! = b_i
    end
    b = zeros(n,1)
    b[k+1] = 1
    return vec(A\b)
end
