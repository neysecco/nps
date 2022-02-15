#This is a collection of optimizers for Python

#IMPORTS
from __future__ import division

###########################################
def fminscg(cost_function, x0, display=True, iterations=1000, grad_tol=1e-10):
    #Scaled conjugate gradient algorithm
    #INPUTS
    #cost_function: function -> scalar function that should be minimized. The function should receive an array as input (EVEN FOR UNIDIMENSIONAL CASES). It should also return two arguments: F (scalar that represents the cost) and dF (array that represents the gradient of F with respect to its inputs).

    #IMPORTS
    from numpy import zeros, sqrt
    from copy import copy

    ##INITIALIZATION
    #STEP 1
    sigma0 = 5e-5
    lambda_k = 5e-7
    lambda_bar_k = 0
    k = 0
    success = True
    keep_looking = True
    N = len(x0)
    x = zeros((N,iterations + 1))
    F = zeros(iterations+1)

    #Preparing for first iteration
    x[:,0] = x0[:]
    F[0], dF = cost_function(x0)
    p_k = -dF[:]
    r_k = -dF[:]
    motive = 'Nothing'

    #Printing
    if display:
        print('====== FMINSCG ======')
        print('FMINSCG Iteration %d | Cost: %4.6e' % (0,F[0]))

    ##MAIN LOOP

    while k < iterations and keep_looking:
        #STEP 2
        if success:
            mod_p_k_2 = p_k.dot(p_k)
            sigma_k = sigma0/sqrt(mod_p_k_2)
            F1, dF1 = cost_function(x[:,k] + sigma_k*p_k)
            s_k = (dF1 - dF)/sigma_k
            delta_k = p_k.dot(s_k)

        #STEP 3
        delta_k = delta_k + (lambda_k - lambda_bar_k)*mod_p_k_2

        #STEP 4
        if delta_k <= 0:
            lambda_bar_k = 2*(lambda_k - delta_k/mod_p_k_2)
            delta_k = -delta_k + lambda_k*mod_p_k_2
            lambda_k = copy(lambda_bar_k)

        #STEP 5
        mi_k = p_k.dot(r_k)
        alpha_k = mi_k/delta_k

        #STEP 6
        x_new = x[:,k] + alpha_k*p_k #NEW VALUE
        F2, dF2 = cost_function(x_new)
        Delta_k = 2*delta_k*(F[k] - F2)/mi_k**2

        #STEP 7
        if Delta_k >= 0:
            x[:,k+1] = x_new[:]
            F[k+1] = F2
            dF = dF2[:]
            r_k_1 = -dF2[:]
            lambda_bar_k = 0
            success = True

            if k % N == 0:
                p_k_1 = r_k_1[:]
            else:
                beta_k = (r_k_1.dot(r_k_1) - r_k_1.dot(r_k))/mi_k
                p_k_1 = r_k_1 + beta_k*p_k

            if Delta_k >= 0.75:
                lambda_k = lambda_k/4

        else:
            x[:,k+1] = x[:,k] #NEW VALUE
            F[k+1] = F[k]
            p_k_1 = p_k[:]
            r_k_1 = r_k[:]
            lambda_bar_k = copy(lambda_k)
            success = False

        #Printing
        if display*((k+1)%50==0):
            print('FMINSCG Iteration %g | Cost: %4.6e' % (k+1,F2))

        #STEP 8
        if Delta_k < 0.25:
            lambda_k = lambda_k + delta_k*(1 - Delta_k)/mod_p_k_2

        #STEP 9
        #Gradient stop
        if r_k_1.dot(r_k_1) >= grad_tol:
            k = k + 1
            p_k = p_k_1[:]
            r_k = r_k_1[:]
        else:
            #Checking if the candidate is a saddle point
            if display:
                print('Checking if candidate is a saddle')
            step = 2*(x[:,k+1] - x[:,k-1]) #Trying to jump over the saddle
            x_jump = x[:,k+1] + step #Analising function after the candidate point
            F_jump, dF_jump = cost_function(x_jump)
            if F_jump <= F[k+1]:
                #The point is probably a saddle point
                if display:
                    print('Jumping saddle!')
                x[:,k+1] = x_jump[:]
                F[k+1] = F_jump
                dF = dF_jump[:]
                k = k + 1
                p_k = p_k_1[:]
                r_k = r_k_1[:]
            else:
                #The point is a minimum/maximum
                if display:
                    print('The candidate is probably an extremum')
                motive = 'Converged: Gradient smaller than grad_tol'
                keep_looking = False

    #End of MAIN LOOP

    #Ending
    if k < iterations:
        x = x[:,:k+2]
        F = F[:k+2]
    else:
        motive = 'Halted: Number of iterations exceeded'

    #Print final message
    if display:
        print(motive)
        print('   End of FMINSCG')
        print('=====================')

    #RETURNS
    return x, F, motive

###########################################
def ALM(design_functions,x0,display=True,major_iterations=50,gamma=1,delta_x_tol=1e-7,grad_opt='fminscg',minor_iterations=1000,grad_tol=1e-10,save_minor_hist=False,forced_iterations=0):
    #INPUTS:
    # design_functions : function handle -> handle for the design functions (objective + constraint functions).
    #                    This function should return F (scalar), dF (1xn array), G (1xm array), dG (mxn array),
    #                    H (1xl array), and dH (lxn array), where F is the scalar objective function value,
    #                    dF is the gradient of the objective function with respect to x, G is the inequality
    #                    constraint functions values, dG is a matrix that contains the gradients of the constraint
    #                    functions with respect to x (Note that all the inequality constraints should be considered in G),
    #                    H is the equality constraint functions values, dH is a matrix that contains the gradients
    #                    of the constraint functions with respect to x (Note that all the equality constraints
    #                    should be considered in this single function). The inequality constraints have the form G < 0,
    #                    and the equality constraints have the form H = 0. When returning multiple gradients
    #                    in a matrix (dG and dH), each column should correspond to a design variable and each row
    #                    corresponds to a constraint function, i.e. each row is a gradient. G and H should ALWAYS arrays,
    #                    even if they have only one element.
    # x0 : 1xn array -> initial design variables vector.
    # grad_opt: string -> Optimizer used for the ALM subproblem (unconstrained optimization).
    #                     Could be either 'fminscg' or any other optimizer from scipy.minimize

    #IMPORTS
    from numpy import ones, zeros, logical_not, abs, max, mod
    from numpy.linalg import norm
    import pickle

    #Get information about initial point
    F0, dF, G0, dG, H0, dH = design_functions(x0)

    #INITIALIZATION
    
    #Initialize Lambdas
    Lambda_G = ones(len(G0))
    Lambda_H = ones(len(H0))

    #Initialize rp
    rp = 1

    #Initialize iteration counter
    k = 0

    #Initialize design variables history
    x = zeros((len(x0),major_iterations+1))
    x[:,0] = x0[:]

    #Initialize design functions history
    F = zeros(major_iterations+1)
    F[0] = F0
    G = zeros((len(G0),major_iterations+1))
    G[:,0] = G0[:]
    H = zeros((len(H0),major_iterations+1))
    H[:,0] = H0[:]

    #Initialize the success flag
    keep_looking = True

    #See if the user requested minor iterations history
    if save_minor_hist:
        from numpy import hstack
        x_minor_hist = [[],[]]

    #PRINTING
    #Print initial message
    if display:
        print('====== ALM ======')
        print('Number of design variables: ',len(x0))
        print('Number of inequality constraints: ',len(G[:,0]))
        print('Number of equality constraints: ',len(H[:,0]))
        print('')
        print('ALM Iteration %g | F: %4.5e max(G): %4.5e max(abs(H)): %4.5e' % (0,F[0],max(G[:,0]),max(abs(H[:,0]))))

    #MAIN LOOP
    while k < major_iterations and keep_looking:

        #Define pseudo-objective function using the new Lambdas
        def cost_function(x):
            return ALM_pseudo_func(x,design_functions,rp,Lambda_G,Lambda_H)

        # Use the appropriate gradient-based optimizer
        if grad_opt.lower() == 'fminscg' or (grad_opt.lower() == 'bfgs' and k < forced_iterations and forced_iterations > 0):

            #Optimize the pseudo-objective function
            x_minor,A,motive = fminscg(cost_function, x[:,k], display, iterations=minor_iterations, grad_tol=grad_tol)

            #Save only the final point
            x[:,k+1] = x_minor[:,-1]

        else:

            from scipy.optimize import minimize

            # Define dummy list to store the gradient to avoid repeated evaluation
            xx = [0]
            dF = [0]
            iter_count = [0]

            def cost_fun(x):
                F, dF_curr = cost_function(x)
                if mod(iter_count[0], 100) == 0:
                    print('iter: %d, obj: '%(iter_count[0]) + str(F))
                xx[0] = x
                dF[0] = dF_curr
                iter_count[0] = iter_count[0] + 1
                return F

            def cost_fun_grad(x):
                # Check if we are asking for the gradient at the same point
                if norm(x - xx[0]) < 1e-10:
                    dF_curr = dF[0]
                else:
                    # need to run a new point
                    F, dF_curr = cost_function(x)
                return dF_curr

            options = {'maxiter':minor_iterations,
                       'gtol':grad_tol}
            res = minimize(cost_fun, x[:,k], method=grad_opt, jac=cost_fun_grad,
                           options=options)
            print(res)
            motive = res.message
            A = res.fun
            x[:,k+1] = res.x

        #Save the minor iteration history if requested by the user
        if save_minor_hist:
            x_minor_hist = hstack([x_minor_hist,x_minor])

        #Compute design functions at the new point
        F[k+1], dF, G[:,k+1], dG, H[:,k+1], dH = design_functions(x[:,k+1])

        #Checking inequality constraints (find Psi)
        G_wins = G[:,k+1] > -Lambda_G/2/rp
        Psi = G_wins*G[:,k+1] + logical_not(G_wins)*(-Lambda_G/2/rp) #Here I only select G where G > -Lambda_G/2/rp

        #Update Lambdas
        Lambda_G = Lambda_G + 2*rp*Psi
        Lambda_H = Lambda_H + 2*rp*H[:,k+1]

        #Update rp
        rp = gamma*rp

        #Print current status
        if display:
             print('ALM Iteration %g | F: %4.5e max(G): %4.5e max(abs(H)): %4.5e' % (k+1,F[k+1],max(G[:,k+1]),max(abs(H[:,k+1]))))

        #Save current history
        with open('ALMhist.pickle','wb') as fid:
            pickle.dump(x[:,:k+1],fid)

        #Check convergence
        if norm(x[:,k+1]-x[:,k]) < delta_x_tol:
            keep_looking = False
        else:
            k = k + 1 #Increment counter and continue iteration

    #Crop preallocated arrays
    if k < major_iterations:
        x = x[:,:k+2]
        F = F[:k+2]
        G = G[:,:k+2]
        H = H[:,:k+2]

    #Print final message
    if display:
        print('     End of ALM')
        print('=====================')

    #RETURNS
    if save_minor_hist: #Return the minor iteration history only if requested by the user
        return x,F,G,H,x_minor_hist
    else:
        return x,F,G,H,[]


###########################################
def ALM_pseudo_func(x,design_functions,rp,Lambda_G,Lambda_H):
    #This function returns the value and the gradient of the pseudo-objective function of the Augmented Lagrangian Method
    #INPUTS:
    #x : 1xn array -> design variables vector.
    #design_functions : function handle -> handle for the design functions (objective + constraint functions). This function should return F (scalar), dF (1xn array), G (1xm array), dG (mxn array), H (1xl array), and dH (lxn array), where F is the scalar objective function value, dF is the gradient of the objective function with respect to x, G is the inequality constraint functions values, dG is a matrix that contains the gradients of the constraint functions with respect to x (Note that all the inequality constraints should be considered in G), H is the equality constraint functions values, dH is a matrix that contains the gradients of the constraint functions with respect to x (Note that all the equality constraints should be considered in this single function). The inequality constraints have the form G < 0, and the equality constraints have the form H = 0. When returning multiple gradients in a matrix (dG and dH), each column should correspond to a design variable and each row corresponds to a constraint function, i.e. each row is a gradient. G and H should ALWAYS arrays, even if they have only one element.
    #rp : scalar -> penalty factor given by the ALM method
    #Lambda_G : array 1xm -> array containing the Lagrange's multipliers for each inequality constraint function
    #Lambda_H : array 1xl -> array containing the Lagrange's multipliers for each equality constraint function

    #IMPORTS
    from numpy import logical_not, tile, array, sum

    #Use the design function to get the values and gradients
    F, dF, G, dG, H, dH = design_functions(x)

    #Checking inequality constraints (find Psi)
    G_wins = G > -Lambda_G/2/rp
    Psi = G_wins*G + logical_not(G_wins)*(-Lambda_G/2/rp) #Here I only select G where G > -Lambda_G/2/rp
    G_wins_matrix = tile(array([G_wins]).T,(1,len(x)))
    dPsi = G_wins_matrix*dG #Save gradients when G > -Lambda_G/2/rp

    #Pseudo-objective function value
    A = F + sum(Lambda_G*Psi + rp*Psi**2) + sum(Lambda_H*H + rp*H**2)

    #Pseudo-objective function gradient
    dPsi = sum(array([Lambda_G+2*rp*Psi]).T*dPsi, axis=0) #Multiplying each gradient by its scalar factor
    dH = sum(array([Lambda_H+2*rp*H]).T*dH, axis=0) #Multiplying each gradient by its scalar factor
    dA = dF + dPsi + dH

    #RETURNS
    return A, dA
