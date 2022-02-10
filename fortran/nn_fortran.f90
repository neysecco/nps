! This is the main subroutine to be called from python

subroutine feedforward(numWeights, numHidLayers, numVars, numCases, &
     Theta_flat, hidLayerTypes, numHidNeurons, X, Y, dYdX, dYdXdX)
  
  use precision ! This imports realKind and intKind
  use baseTypes ! This imports type(layerType)
  use steps ! This imports feedforward_step
  implicit none

  ! Input variables
  integer(kind=intKind), intent(in) :: numWeights, numHidLayers, numVars, numCases
  real(kind=realKind), dimension(numWeights), intent(in) :: Theta_flat
  integer(kind=intKind), dimension(numHidLayers), intent(in) :: hidLayerTypes, numHidNeurons
  real(kind=realKind), dimension(numVars, numCases), intent(in) :: X
  !f2py intent(in) numWeights, numHidLayers, numVars, numCases
  !f2py intent(in) Theta_flat, hidLayerTypes, numHidNeurons, X

  ! Output variables
  real(kind=realKind), dimension(numCases), intent(out) :: Y
  real(kind=realKind), dimension(numVars, numCases), intent(out) :: dYdX
  real(kind=realKind), dimension(numVars, numVars, numCases), intent(out) :: dYdXdX
  !f2py intent(out) Y, dYdX, dYdXdX

  ! Working variables
  type(layerType), dimension(0:numHidLayers+1) :: layer
  integer(kind=intKind) :: theta_start, theta_end, currWeights, index

  ! BEGIN EXECUTION

  ! Allocate all layers (allocateNN is defined in baseTypes.f90)
  call allocateNN(Theta_flat, numHidLayers, hidLayerTypes, numHidNeurons, X, layer)

  ! Now do the feedforward step for each layer
  do index = 1,numHidLayers+1
     call feedforward_step(layer(index-1)%numNeurons, layer(index)%numNeurons, numVars, numCases, &
          layer(index)%transFuncType, layer(index)%Theta, layer(index-1)%Act, &
          layer(index-1)%dActdX, layer(index-1)%dActdXdX, layer(index)%Act, layer(index)%dActdX, &
          layer(index)%dActdXdX, layer(index)%phiZ, layer(index)%phiDZ, layer(index)%phiDDZ, &
          layer(index)%phiDDDZ,  layer(index)%dZdX, layer(index)%dZdXdX)
  end do

  ! Assign ANN outputs
  Y = layer(numHidLayers+1)%Act(1,:)
  dYdX = layer(numHidLayers+1)%dActdX(:,1,:)
  dYdXdX = layer(numHidLayers+1)%dActdXdX(:,:,1,:)

end subroutine feedforward

!===============================================================

subroutine backpropagation(numWeights, numHidLayers, numVars, numCases, &
     Theta_flat, hidLayerTypes, numHidNeurons, X, Y, dYdX, dYdXdX, &
     dYdTheta_flat, dYdXdTheta_flat, dYdXdXdTheta_flat)
  
  use precision ! This imports realKind and intKind
  use baseTypes ! This imports type(layerType)
  use steps ! This imports feedforward_step and backpropagation_step
  implicit none

  ! Input variables
  integer(kind=intKind), intent(in) :: numWeights, numHidLayers, numVars, numCases
  real(kind=realKind), dimension(numWeights), intent(in) :: Theta_flat
  integer(kind=intKind), dimension(numHidLayers), intent(in) :: hidLayerTypes, numHidNeurons
  real(kind=realKind), dimension(numVars, numCases), intent(in) :: X
  !f2py intent(in) numWeights, numHidLayers, numVars, numCases
  !f2py intent(in) Theta_flat, hidLayerTypes, numHidNeurons, X

  ! Output variables
  real(kind=realKind), dimension(numCases), intent(out) :: Y
  real(kind=realKind), dimension(numVars, numCases), intent(out) :: dYdX
  real(kind=realKind), dimension(numVars, numVars, numCases), intent(out) :: dYdXdX
  real(kind=realKind), dimension(numWeights, numCases), intent(out) :: dYdTheta_flat
  real(kind=realKind), dimension(numWeights, numVars, numCases), intent(out) :: dYdXdTheta_flat
  real(kind=realKind), dimension(numWeights, numVars, numVars, numCases), intent(out) :: dYdXdXdTheta_flat
  !f2py intent(out) Y, dYdX, dYdXdX, dYdTheta_flat, dYdXdTheta_flat, dYdXdXdTheta_flat

  ! Working variables
  type(layerType), dimension(0:numHidLayers+1) :: layer
  integer(kind=intKind) :: theta_start, theta_end, currWeights, index

  ! BEGIN EXECUTION

  ! Allocate all layers (allocateNN is defined in baseTypes.f90)
  call allocateNN(Theta_flat, numHidLayers, hidLayerTypes, numHidNeurons, X, layer)

  ! Now do the feedforward step for each layer
  do index = 1,numHidLayers+1
     call feedforward_step(layer(index-1)%numNeurons, layer(index)%numNeurons, numVars, numCases, &
          layer(index)%transFuncType, layer(index)%Theta, layer(index-1)%Act, &
          layer(index-1)%dActdX, layer(index-1)%dActdXdX, layer(index)%Act, layer(index)%dActdX, &
          layer(index)%dActdXdX, layer(index)%phiZ, layer(index)%phiDZ, layer(index)%phiDDZ, &
          layer(index)%phiDDDZ,  layer(index)%dZdX, layer(index)%dZdXdX)
  end do

  ! Assign ANN outputs
  Y = layer(numHidLayers+1)%Act(1,:)
  dYdX = layer(numHidLayers+1)%dActdX(:,1,:)
  dYdXdX = layer(numHidLayers+1)%dActdXdX(:,:,1,:)

  ! Initialize sensitivities of the last layer
  layer(numHidLayers+1)%dYdAct = 1.0
  layer(numHidLayers+1)%dYdXdAct = 0.0
  layer(numHidLayers+1)%dYdXdXdAct = 0.0

  ! Initialize counter to slice Theta_flat
  theta_end = numWeights

  ! Now it is time to backpropagate
  do index = NumHidLayers+1,1,-1

     call backpropagation_step(layer(index-1)%numNeurons, layer(index)%numNeurons, numVars, &
          numCases, layer(index)%Theta, layer(index-1)%Act, layer(index-1)%dActdX, &
          layer(index-1)%dActdXdX, layer(index)%phiZ, layer(index)%phiDZ, layer(index)%phiDDZ, &
          layer(index)%phiDDDZ,  layer(index)%dZdX, layer(index)%dZdXdX, &
          layer(index)%dYdAct, layer(index)%dYdXdAct, layer(index)%dYdXdXdAct, &
          layer(index)%dYdTheta, layer(index)%dYdXdTheta, layer(index)%dYdXdXdTheta, &
          layer(index-1)%dYdAct, layer(index-1)%dYdXdAct, layer(index-1)%dYdXdXdAct)

     ! Find final position of the Theta_flat array
     currWeights = layer(index)%numNeurons*(layer(index-1)%numNeurons+1)
     theta_start = theta_end - currWeights + 1

     ! Then we need to save the weights derivatives
     dYdTheta_flat(theta_start:theta_end,:) = reshape(layer(index)%dYdTheta, &
          (/currWeights,numCases/))
     dYdXdTheta_flat(theta_start:theta_end,:,:) = reshape(layer(index)%dYdXdTheta, &
          (/currWeights,numVars,numCases/),order=(/2,1,3/))
     dYdXdXdTheta_flat(theta_start:theta_end,:,:,:) = reshape(layer(index)%dYdXdXdTheta, &
          (/currWeights,numVars,numVars,numCases/),order=(/3,2,1,4/))

     ! Reassing auxiliary variable
     theta_end = theta_start - 1

  end do

end subroutine backpropagation
