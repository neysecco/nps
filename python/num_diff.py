def center_fd(function, X0, step, *args):

        #Each input case is a column

        #Imports
        from numpy import shape, zeros, array

        #Checking the dimension of the problem
        (n,m) = shape(X0) #Number of variables and number of cases
        F_X0 = function(X0, *args) #Sampling the given point
        (t,m2) = shape(F_X0) #Find number of outputs and number of cases
        if m != m2: #Checking consistency between inputs and outputs
                print('Number of cases is inconsistent between inputs and outputs')
                print('Here is the correct formating')
                print('X0 : (num_inputs x num_cases)')
                print('F : (num_outputs x num_cases)')
                exit()

        #Initializing gradF
        gradF = zeros((shape(F_X0)[0],shape(X0)[0],shape(X0)[1]))

        for i in range(0,n):
                X = array(X0)
                X[i,:] = X0[i,:] + step
                F_Xup = function(X, *args)
                X[i,:] = X0[i,:] - step
                F_Xdown = function(X, *args)
                gradF[:,i,:] = (F_Xup - F_Xdown)/2/step

        return gradF

###########################################

def complex_step(function, X0, step, *args):

        #Each input case is a column

        #Imports
        from numpy import shape, zeros, array

        #Checking the dimension of the problem
        (n,m) = shape(X0) #Number of variables and number of cases
        F_X0 = function(X0, *args) #Sampling the given point
        (t,m2) = shape(F_X0) #Find number of outputs and number of cases
        if m != m2: #Checking consistency between inputs and outputs
                print('Number of cases is inconsistent between inputs and outputs')
                print('Here is the correct formating')
                print('X0 : (num_inputs x num_cases)')
                print('F : (num_outputs x num_cases)')
                exit()

        #Initializing gradF
        gradF = zeros((shape(F_X0)[0],shape(X0)[0],shape(X0)[1]))

        for i in range(0,n):
                X = array(X0,dtype=complex)
                X[i,:] = X0[i,:] + step*1.0j
                F_comp = function(X, *args)
                gradF[:,i,:] = F_comp.imag/step

        return gradF

###########################################

def complex_step_par(function, X0, step, *args):

        #Each input case is a column

        #Imports
        from numpy import shape, zeros, array
        import multiprocessing as mp
        #import shmarray

        #Checking the dimension of the problem
        (n,m) = shape(X0) #Number of variables and number of cases
        F_X0 = function(X0, *args) #Sampling the given point
        (t,m2) = shape(F_X0) #Find number of outputs and number of cases
        if m != m2: #Checking consistency between inputs and outputs
                print('Number of cases is inconsistent between inputs and outputs')
                print('Here is the correct formating')
                print('X0 : (num_inputs x num_cases)')
                print('F : (num_outputs x num_cases)')
                exit()

        #Initializing gradF
        gradF = zeros((shape(F_X0)[0],shape(X0)[0],shape(X0)[1]))

        def start_process():
                print('Starting ',mp.current_process().name)

        #Initialize pool
        pool_size = mp.cpu_count()
        pool = mp.Pool(processes=pool_size, initializer=start_process)

        #Function that will be parallelized
        def partial_derivative(i): #This function gives the partial derivative for the variable i
                X = array(X0,dtype=complex)
                X[i,:] = X0[i,:] + step*1.0j
                F_comp = function(X, *args)
                gradF[:,i,:] =  F_comp.imag/step

        '''
        #Start multiprocessing
        procs = [Process(target=partial_derivative, args=(i,)) for i in range(0,n)]

        #Execute in parallel
        for p in procs:
                p.start()
        for p in procs:
                p.join()
        #map(lambda x: x.start(), procs)
        #map(lambda x: x.join(), procs)
        '''

        pool.map(partial_derivative, range(0,n))
        pool.close()

        return gradF
