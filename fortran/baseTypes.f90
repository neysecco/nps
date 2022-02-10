module baseTypes

  use precision
  implicit none

  type layerType

     ! This type hold information about a specific layer

     integer(kind=intKind) :: transFuncType, numNeurons

     real(kind=realKind), dimension(:,:), allocatable :: Theta, Act
     real(kind=realKind), dimension(:,:), allocatable :: phiZ, phiDZ, phiDDZ, phiDDDZ
     real(kind=realKind), dimension(:,:,:), allocatable :: dActdX, dZdX
     real(kind=realKind), dimension(:,:,:,:), allocatable :: dActdXdX, dZdXdX

     real(kind=realKind), dimension(:,:), allocatable :: dYdAct
     real(kind=realKind), dimension(:,:,:), allocatable :: dYdXdAct
     real(kind=realKind), dimension(:,:,:,:), allocatable :: dYdXdXdAct
     real(kind=realKind), dimension(:,:,:), allocatable :: dYdTheta
     real(kind=realKind), dimension(:,:,:,:), allocatable :: dYdXdTheta
     real(kind=realKind), dimension(:,:,:,:,:), allocatable :: dYdXdXdTheta

  end type layerType

contains

  subroutine allocateNN(Theta_flat, numHidLayers, hidLayerTypes, numHidNeurons, X, layer)

    ! This subroutine allocates the neural network and assigns the weights
    ! matrices to each layer

    ! Theta_flat: 1D array with all NN weights
    ! hidLayerTypes: 1D array of integers indicating transfer functions

    use precision
    implicit none

    ! Input variables
    real(kind=realKind), dimension(:), intent(in) :: Theta_flat
    integer(kind=intKind), intent(in) :: numHidLayers
    integer(kind=intKind), dimension(:), intent(in) :: hidLayerTypes, numHidNeurons
    real(kind=realKind), dimension(:,:), intent(in) :: X

    ! Output variables
    type(layerType), dimension(0:numHidLayers+1), intent(out) :: layer

    ! Working variables
    integer(kind=intKind) :: numCases, numVars, index
    integer(kind=intKind) :: theta_start, theta_end
    integer(kind=intKind) :: prevNumNeurons, currNumNeurons

    ! BEGIN EXECUTION

    ! Determine problem size
    numVars = size(X, 1)
    numCases = size(X, 2)

    ! Allocate input layer
    allocate(layer(0)%Act(numVars, numCases))
    allocate(layer(0)%dActdX(numVars, numVars, numCases))
    allocate(layer(0)%dActdXdX(numVars, numVars, numVars, numCases))
    allocate(layer(0)%dYdAct(numVars, numCases))
    allocate(layer(0)%dYdXdAct(numVars, numVars, numCases))
    allocate(layer(0)%dYdXdXdAct(numVars, numVars, numVars, numCases))

    ! Make assignments to input layer
    layer(0)%numNeurons = numVars
    layer(0)%Act = X
    layer(0)%dActdX = 0.0
    do index = 1,numVars
       layer(0)%dActdX(index, index, :) = 1.0
    end do
    layer(0)%dActdXdX = 0.0

    ! Allocate hidden layers
    theta_start = 1 ! Initialize counter to split Theta_flat
    prevNumNeurons = numVars ! Store number of neurons in layer 0
    do index = 1,numHidLayers

       ! Get current number of neurons
       currNumNeurons = numHidNeurons(index)

       allocate(layer(index)%Theta(currNumNeurons, prevNumNeurons+1))
       allocate(layer(index)%Act(currNumNeurons,numCases))
       allocate(layer(index)%phiZ(currNumNeurons,numCases))
       allocate(layer(index)%phiDZ(currNumNeurons,numCases))
       allocate(layer(index)%phiDDZ(currNumNeurons,numCases))
       allocate(layer(index)%phiDDDZ(currNumNeurons,numCases))
       allocate(layer(index)%dActdX(numVars,currNumNeurons,numCases))
       allocate(layer(index)%dZdX(numVars,currNumNeurons,numCases))
       allocate(layer(index)%dActdXdX(numVars,numVars,currNumNeurons,numCases))
       allocate(layer(index)%dZdXdX(numVars,numVars,currNumNeurons,numCases))
       allocate(layer(index)%dYdAct(currNumNeurons, numCases))
       allocate(layer(index)%dYdXdAct(numVars, currNumNeurons, numCases))
       allocate(layer(index)%dYdXdXdAct(numVars, numVars, currNumNeurons, numCases))
       allocate(layer(index)%dYdTheta(currNumNeurons, prevNumNeurons+1, numCases))
       allocate(layer(index)%dYdXdTheta(numVars, currNumNeurons, prevNumNeurons+1, numCases))
       allocate(layer(index)%dYdXdXdTheta(numVars, numVars, currNumNeurons, prevNumNeurons+1, numCases))

       ! Assign number of neurons
       layer(index)%numNeurons = currNumNeurons

       ! Assign type
       layer(index)%transFuncType = hidLayerTypes(index)

       ! Assign weights
       theta_end = theta_start + currNumNeurons*(prevNumNeurons+1) - 1
       layer(index)%Theta = reshape(Theta_flat(theta_start:theta_end), &
            (/currNumNeurons, prevNumNeurons+1/))
       theta_start = theta_end + 1

       ! Update number of neurons for next iteration
       prevNumNeurons = currNumNeurons

    end do

    ! Get number of neurons of the last layer (assuming single output)
    currNumNeurons = 1

    ! Allocate last layer
    index = numHidLayers+1
    allocate(layer(index)%Theta(currNumNeurons, prevNumNeurons+1))
    allocate(layer(index)%Act(currNumNeurons,numCases))
    allocate(layer(index)%phiZ(currNumNeurons,numCases))
    allocate(layer(index)%phiDZ(currNumNeurons,numCases))
    allocate(layer(index)%phiDDZ(currNumNeurons,numCases))
    allocate(layer(index)%phiDDDZ(currNumNeurons,numCases))
    allocate(layer(index)%dActdX(numVars,currNumNeurons,numCases))
    allocate(layer(index)%dZdX(numVars,currNumNeurons,numCases))
    allocate(layer(index)%dActdXdX(numVars,numVars,currNumNeurons,numCases))
    allocate(layer(index)%dZdXdX(numVars,numVars,currNumNeurons,numCases))
    allocate(layer(index)%dYdAct(currNumNeurons, numCases))
    allocate(layer(index)%dYdXdAct(numVars, currNumNeurons, numCases))
    allocate(layer(index)%dYdXdXdAct(numVars, numVars, currNumNeurons, numCases))
    allocate(layer(index)%dYdTheta(currNumNeurons, prevNumNeurons+1, numCases))
    allocate(layer(index)%dYdXdTheta(numVars, currNumNeurons, prevNumNeurons+1, numCases))
    allocate(layer(index)%dYdXdXdTheta(numVars, numVars, currNumNeurons, prevNumNeurons+1, numCases))

    ! Assign number of neurons
    layer(index)%numNeurons = currNumNeurons

    ! Assign weights
    theta_end = theta_start + currNumNeurons*(prevNumNeurons+1) - 1
    layer(index)%Theta = reshape(Theta_flat(theta_start:theta_end), &
         (/currNumNeurons, prevNumNeurons+1/))

    ! Assign linear transfer function (type = 1) to the output layer
    layer(index)%transFuncType = 1

    ! Check if we used all elements of Theta_flat (ONLY FOR DEBUGGING)
    !if (theta_end .ne. size(Theta_flat)) then
       !print *,'Did not use full Theta_flat'
       !stop
    !end if

  end subroutine allocateNN

end module baseTypes
