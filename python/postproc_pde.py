def plot_residuals(NN_set, Residual_BC, Residual_PDE, Theta_flat_hist, Theta_step):

    # IMPORTS
    from numpy import zeros
    from .funcs_pde import reassign_theta
    import matplotlib.pyplot as plt

    # Get the number of iterations
    num_iter = Theta_flat_hist.shape[1]

    # Get the number of BC points and PDE points
    num_BC = len(Residual_BC(NN_set)[0])
    num_PDE = len(Residual_PDE(NN_set)[0])

    # Get the number of weights
    num_weights = len(Theta_flat_hist[:,0])

    #Create array with relevant iteration indices
    iter_list = range(0,num_iter,Theta_step)

    # Allocate Residuals arrays
    ResBC = zeros(len(iter_list))
    ResPDE = zeros(len(iter_list))
    Reg = zeros(len(iter_list))

    # Loop over each iteration weight
    for index in range(len(iter_list)):

        # Reassign weights
        reassign_theta(NN_set, Theta_flat_hist[:,iter_list[index]])

        # Run residuals code (discard the gradients)
        ResBC[index] = sum(Residual_BC(NN_set)[0]**2)/2/num_BC
        ResPDE[index] = sum(Residual_PDE(NN_set)[0]**2)/2/num_PDE
        Reg[index] = sum(Theta_flat_hist[:,iter_list[index]]**2)/2/num_weights

    # Plot residuals
    fig = plt.figure()
    plt.semilogy(iter_list, ResBC, linewidth=2, label='BC Residuals')
    plt.semilogy(iter_list, ResPDE, linewidth=2, label='PDE Residuals')
    plt.semilogy(iter_list, Reg, linewidth=2, label='ANN Weights')
    plt.xlabel('Major ALM iterations',fontsize=16)
    plt.ylabel('Mean Squared Value',fontsize=16)
    plt.legend(loc='best',fontsize=14)

    # Save figure
    fig.savefig('residuals.pdf',dpi=300)

    # Return figure handle and residuals history
    return fig, ResBC, ResPDE, Reg

###########################################

###########################################

def plot_video(NN_set, plot_function, Theta_flat_hist, images_dir, fps, Theta_step):

    # IMPORTS
    import os
    import matplotlib.pyplot as plt
    from .funcs_pde import reassign_theta

    # Get the number of iterations
    num_iter = Theta_flat_hist.shape[1]

    # Initialize figure counter (will be used to write the figures file names)
    counter = 0

    # Remove all images from the folder
    import os
    os.system('rm '+images_dir+'*.png')

    # Loop over each iteration weight
    for index in range(0,num_iter,Theta_step):

        # Reassign weights
        reassign_theta(NN_set, Theta_flat_hist[:,index])

        # Call plotting function
        fig = plot_function(NN_set)

        # Save figure
        fig.savefig(images_dir+'img_%05d.png'%counter,dpi=100)

        # Increment figure counter
        counter = counter + 1

        # Close figure
        plt.close()

    # Generate video
    os.system('ffmpeg -f image2 -r ' + str(fps) + ' -i '+images_dir+'img_%05d.png -vcodec mpeg4 -y movie.mp4')

###########################################

###########################################
