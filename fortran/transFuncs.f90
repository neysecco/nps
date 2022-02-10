module transFuncs

contains
  
  subroutine applyTransFunc(numNeurons, numCases, Z, layerType, &
       phiZ, phiDZ, phiDDZ, phiDDDZ)

    ! This subroutine applies a transfer function to Z
    ! based on the layerType integer. The following types are
    ! supported:
    ! 1: linear
    ! 2: sigmoid

    use precision ! This brings intKind and realKind
    implicit none

    ! Input variables
    integer(kind=intKind), intent(in) :: numNeurons, numCases
    real(kind=realKind), dimension(numNeurons,numCases), intent(in) :: Z
    integer(kind=intKind), intent(in) :: layerType
    
    ! Output variables
    real(kind=realKind), dimension(numNeurons,numCases), intent(out) :: phiZ, phiDZ, phiDDZ, phiDDDZ

    ! BEGIN EXECUTION
    
    ! Apply transfer function based on its type
    if (layerType .eq. 1) then
       call linear(numNeurons, numCases, Z, phiZ, phiDZ, phiDDZ, phiDDDZ)
    else if (layerType .eq. 2) then
       call sigmoid(numNeurons, numCases, Z, phiZ, phiDZ, phiDDZ, phiDDDZ)
    else
       print *,'transfer function not supported'
       stop
    end if

  end subroutine applyTransFunc

  subroutine linear(numNeurons, numCases, Z, phiZ, phiDZ, phiDDZ, phiDDDZ)

    ! This is the linear function
    ! phi(Z) = Z

    use precision ! This brings intKind and realKind
    implicit none

    ! Input variables
    integer(kind=intKind), intent(in) :: numNeurons, numCases
    real(kind=realKind), dimension(numNeurons,numCases), intent(in) :: Z

    ! Output variables
    real(kind=realKind), dimension(numNeurons,numCases), intent(out) :: phiZ, phiDZ, phiDDZ, phiDDDZ

    ! BEGIN EXECUTION
    phiZ = Z
    phiDZ = 1.0
    phiDDZ = 0.0
    phiDDDZ = 0.0
    
  end subroutine linear

  subroutine sigmoid(numNeurons, numCases, Z, phiZ, phiDZ, phiDDZ, phiDDDZ)

    ! This is the sigmoid function
    ! phi(Z) = 2.0/(1.0 + exp(-2.0*Z)) - 1.0

    use precision ! This brings intKind and realKind
    implicit none

    ! Input variables
    integer(kind=intKind), intent(in) :: numNeurons, numCases
    real(kind=realKind), dimension(numNeurons,numCases), intent(in) :: Z

    ! Output variables
    real(kind=realKind), dimension(numNeurons,numCases), intent(out) :: phiZ, phiDZ, phiDDZ, phiDDDZ

    ! BEGIN EXECUTION
    phiZ = 2.0/(1.0 + exp(-2.0*Z)) - 1.0
    phiDZ = 1.0 - phiZ*phiZ
    phiDDZ = -2.0*phiZ*phiDZ
    phiDDDZ = -2.0*(phiDZ*phiDZ + phiZ*phiDDZ)
    
  end subroutine sigmoid

end module transFuncs
