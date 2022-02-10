module steps

! This module contains the equations for layer-by-layer
! propagation

contains
  
  subroutine feedforward_step(numNeurons0, numNeurons1, numVars, numCases, layerType, &
       Theta, Act0, dAct0dX, dAct0dXdX, Act1, dAct1dX, dAct1dXdX, phiZ1, phiDZ1, phiDDZ1, phiDDDZ1, &
       dZ1dX, dZ1dXdX)

    use precision ! this brings intType and realKind
    use transFuncs ! this brings applyTransFunc
    implicit none

    ! Input variables
    integer(kind=intKind), intent(in) :: numNeurons0, numNeurons1, numVars, numCases, layerType
    real(kind=realKind), dimension(numNeurons1, numNeurons0+1), intent(in) :: Theta
    real(kind=realKind), dimension(numNeurons0, numCases), intent(in) :: Act0
    real(kind=realKind), dimension(numVars, numNeurons0, numCases), intent(in) :: dAct0dX
    real(kind=realKind), dimension(numVars, numVars, numNeurons0, numCases), intent(in) :: dAct0dXdX

    ! Output variables
    real(kind=realKind), dimension(numNeurons1, numCases), intent(out) :: Act1
    real(kind=realKind), dimension(numVars, numNeurons1, numCases), intent(out) :: dAct1dX
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numCases), intent(out) :: dAct1dXdX
    real(kind=realKind), dimension(numNeurons1, numCases), intent(out) :: phiZ1, phiDZ1, phiDDZ1, phiDDDZ1
    real(kind=realKind), dimension(numVars, numNeurons1, numCases), intent(out) :: dZ1dX
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numCases), intent(out) :: dZ1dXdX

    ! Working variables
    real(kind=realKind), dimension(numNeurons1, numCases) :: Z1
    integer(kind=intKind) :: i, j, k, p, q

    ! BEGIN EXECUTION

    ! Initialize variables
    Act1 = 0
    dAct1dX = 0
    dAct1dXdX = 0
    Z1 = 0
    dZ1dX = 0
    dZ1dXdX = 0

    ! Loop
    do k = 1,numCases
       do p = 1,numNeurons1

          ! Bias equation
          Z1(p,k) = Z1(p,k) + Theta(p,1) ! Eq. 56

          do q = 1,numNeurons0
             Z1(p,k) = Z1(p,k) + Theta(p,q+1)*Act0(q,k) ! Eq. 56
             do j = 1,numVars
                dZ1dX(j,p,k) = dZ1dX(j,p,k) + Theta(p,q+1)*dAct0dX(j,q,k) ! Eq. 58
                do i = 1,j
                   dZ1dXdX(i,j,p,k) = dZ1dXdX(i,j,p,k) + Theta(p,q+1)*dAct0dXdX(i,j,q,k) ! Eq. 60
                   dZ1dXdX(j,i,p,k) = dZ1dXdX(i,j,p,k) ! Symmetric
                end do
             end do
          end do
       end do
    end do

    ! Apply transfer functions all at once to avoid multiple function calls
    ! This subroutine is defined in transFuncs.f90
    call applyTransFunc(numNeurons1, numCases, Z1, layerType, phiZ1, phiDZ1, &
         phiDDZ1, phiDDDZ1)

    ! Second loop (after we have transfer functions)
    do k = 1,numCases
       do p = 1,numNeurons1
          Act1(p,k) = phiZ1(p,k) ! Eq. 57
          do j = 1,numVars
             dAct1dX(j,p,k) = phiDZ1(p,k)*dZ1dX(j,p,k) ! Eq. 59
             do i = 1,j
                dAct1dXdX(i,j,p,k) = phiDDZ1(p,k)*dZ1dX(i,p,k)*dZ1dX(j,p,k) + &
                     phiDZ1(p,k)*dZ1dXdX(i,j,p,k) ! Eq. 61
                dAct1dXdX(j,i,p,k) = dAct1dXdX(i,j,p,k) ! Symmetric
             end do
          end do
       end do
    end do

  end subroutine feedforward_step

  subroutine backpropagation_step(numNeurons0, numNeurons1, numVars, numCases, &
       Theta, Act0, dAct0dX, dAct0dXdX, phiZ1, phiDZ1, phiDDZ1, phiDDDZ1, dZ1dX, dZ1dXdX, &
       dYdAct1, dYdXdAct1, dYdXdXdAct1, dYdTheta, dYdXdTheta, dYdXdXdTheta, &
       dYdAct0, dYdXdAct0, dYdXdXdAct0)

    use precision
    implicit none

    ! Input variables
    integer(kind=intKind), intent(in) :: numNeurons0, numNeurons1, numVars, numCases
    real(kind=realKind), dimension(numNeurons1, numNeurons0+1), intent(in) :: Theta
    real(kind=realKind), dimension(numNeurons0, numCases), intent(in) :: Act0
    real(kind=realKind), dimension(numVars, numNeurons0, numCases), intent(in) :: dAct0dX
    real(kind=realKind), dimension(numVars, numVars, numNeurons0, numCases), intent(in) :: dAct0dXdX
    real(kind=realKind), dimension(numNeurons1, numCases), intent(in) :: phiZ1, phiDZ1, phiDDZ1, phiDDDZ1
    real(kind=realKind), dimension(numVars, numNeurons1, numCases), intent(in) :: dZ1dX
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numCases), intent(in) :: dZ1dXdX
    real(kind=realKind), dimension(numNeurons1, numCases), intent(in) :: dYdAct1
    real(kind=realKind), dimension(numVars, numNeurons1, numCases), intent(in) :: dYdXdAct1
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numCases), intent(in) :: dYdXdXdAct1

    ! Output variables
    real(kind=realKind), dimension(numNeurons1, numNeurons0+1, numCases), intent(out) :: dYdTheta
    real(kind=realKind), dimension(numVars, numNeurons1, numNeurons0+1, numCases), intent(out) :: dYdXdTheta
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numNeurons0+1, numCases), intent(out) :: dYdXdXdTheta
    real(kind=realKind), dimension(numNeurons0, numCases), intent(out) :: dYdAct0
    real(kind=realKind), dimension(numVars, numNeurons0, numCases), intent(out) :: dYdXdAct0
    real(kind=realKind), dimension(numVars, numVars, numNeurons0, numCases), intent(out) :: dYdXdXdAct0

    ! Working variables
    real(kind=realKind), dimension(numNeurons1, numNeurons0+1, numCases) :: dAct1dTheta
    real(kind=realKind), dimension(numVars, numNeurons1, numNeurons0+1, numCases) :: dAct1dXdTheta
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numNeurons0+1, numCases) :: dAct1dXdXdTheta
    real(kind=realKind), dimension(numNeurons1, numNeurons0, numCases) :: dAct1dAct0
    real(kind=realKind), dimension(numVars, numNeurons1, numNeurons0, numCases) :: dAct1dXdAct0
    real(kind=realKind), dimension(numVars, numVars, numNeurons1, numNeurons0, numCases) :: dAct1dXdXdAct0
    integer(kind=intKind) :: i, j, k, p, q

    ! BEGIN EXECUTION

    do k = 1,numCases
       do p = 1,numNeurons1

          ! Bias equations
          dAct1dTheta(p,1,k) = phiDZ1(p,k) ! Eq. 68
          do j = 1,numVars
             dAct1dXdTheta(j,p,1,k) = phiDDZ1(p,k)*dZ1dX(j,p,k) ! Eq. 69
             do i = 1,j
                dAct1dXdXdTheta(i,j,p,1,k) = phiDDDZ1(p,k)*dZ1dX(i,p,k)*dZ1dX(j,p,k) + &
                     PhiDDZ1(p,k)*dZ1dXdX(i,j,p,k) ! Eq. 70
                dAct1dXdXdTheta(j,i,p,1,k) = dAct1dXdXdTheta(i,j,p,1,k) ! Symmetric
             end do
          end do

          do q = 1,numNeurons0
             dAct1dTheta(p,q+1,k) = phiDZ1(p,k)*Act0(q,k) ! Eq. 68
             dAct1dAct0(p,q,k) = phiDZ1(p,k)*Theta(p,q+1) ! Eq. 71
             do j = 1,numVars
                dAct1dXdTheta(j,p,q+1,k) = phiDDZ1(p,k)*Act0(q,k)*dZ1dX(j,p,k) + phiDZ1(p,k)*dAct0dX(j,q,k) ! Eq. 69
                dAct1dXdAct0(j,p,q,k) = phiDDZ1(p,k)*Theta(p,q+1)*dZ1dX(j,p,k) ! Eq. 72
                do i = 1,j
                   dAct1dXdXdTheta(i,j,p,q+1,k) = phiDDDZ1(p,k)*Act0(q,k)*dZ1dX(i,p,k)*dZ1dX(j,p,k) + &
                        PhiDDZ1(p,k)*(dAct0dX(i,q,k)*dZ1dX(j,p,k) + dZ1dX(i,p,k)*dAct0dX(j,q,k) + &
                        Act0(q,k)*dZ1dXdX(i,j,p,k)) + phiDZ1(p,k)*dAct0dXdX(i,j,q,k) ! Eq. 70
                   dAct1dXdXdTheta(j,i,p,q+1,k) = dAct1dXdXdTheta(i,j,p,q+1,k) ! Symmetric
                   dAct1dXdXdAct0(i,j,p,q,k) = Theta(p,q+1)*(phiDDDZ1(p,k)*dZ1dX(i,p,k)*dZ1dX(j,p,k) + &
                        phiDDZ1(p,k)*dZ1dXdX(i,j,p,k)) ! Eq. 73
                   dAct1dXdXdAct0(j,i,p,q,k) = dAct1dXdXdAct0(i,j,p,q,k) ! Symmetric
                end do
             end do
          end do
       end do
    end do

    ! Computing weights derivatives
    do k = 1,numCases
       do q = 1,numNeurons0+1
          do p = 1,numNeurons1
             dYdTheta(p,q,k) = dYdAct1(p,k)*dAct1dTheta(p,q,k) ! Eq. 74
             do j = 1,numVars
                dYdXdTheta(j,p,q,k) = dYdXdAct1(j,p,k)*dAct1dTheta(p,q,k) + &
                     dYdAct1(p,k)*dAct1dXdTheta(j,p,q,k) ! Eq. 75
                do i = 1,j
                   dYdXdXdTheta(i,j,p,q,k) = dYdXdXdAct1(i,j,p,k)*dAct1dTheta(p,q,k) + &
                        dYdXdAct1(j,p,k)*dAct1dXdTheta(i,p,q,k) + dYdXdAct1(i,p,k)*dAct1dXdTheta(j,p,q,k) + &
                        dYdAct1(p,k)*dAct1dXdXdTheta(i,j,p,q,k) ! Eq. 76
                   dYdXdXdTheta(j,i,p,q,k) = dYdXdXdTheta(i,j,p,q,k) ! Symmetric
                end do
             end do
          end do
       end do
    end do

    ! Initialize variables
    dYdAct0 = 0.0
    dYdXdAct0 = 0.0
    dYdXdXdAct0 = 0.0

    ! Backpropagating activation sensitivities
    do k = 1,numCases
       do q = 1,numNeurons0
          do p = 1,numNeurons1
             dYdAct0(q,k) = dYdAct0(q,k) + dYdAct1(p,k)*dAct1dAct0(p,q,k) ! Eq. 77
             do j = 1,numVars
                dYdXdAct0(j,q,k) = dYdXdAct0(j,q,k) + dYdXdAct1(j,p,k)*dAct1dAct0(p,q,k) + &
                     dYdAct1(p,k)*dAct1dXdAct0(j,p,q,k) ! Eq. 78
                do i = 1,j
                   dYdXdXdAct0(i,j,q,k) = dYdXdXdAct0(i,j,q,k) + dYdXdXdAct1(i,j,p,k)*dAct1dAct0(p,q,k) + &
                        dYdXdAct1(j,p,k)*dAct1dXdAct0(i,p,q,k) + dYdXdAct1(i,p,k)*dAct1dXdAct0(j,p,q,k) + &
                        dYdAct1(p,k)*dAct1dXdXdAct0(i,j,p,q,k) ! Eq. 79
                   dYdXdXdAct0(j,i,q,k) = dYdXdXdAct0(i,j,q,k) ! Symmetric
                end do
             end do
          end do
       end do
    end do

  end subroutine backpropagation_step

end module steps
