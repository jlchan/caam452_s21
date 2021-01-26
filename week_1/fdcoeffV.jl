function fdcoeffV(k,x̄,x)
    n = length(x)
    if n <= k
        error("Too few points in the stencil")
    end
    A = ones(n,n)
    xrow = (x .- x̄)
    for i = 2:n
        A[i,:] = @. xrow ^(i-1) / factorial(i-1)
    end
    b = zeros(n,1)
    b[k+1] = 1
    return vec(A\b)
end
