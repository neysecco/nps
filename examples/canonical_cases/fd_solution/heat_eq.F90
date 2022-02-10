subroutine gauss_seidel(n,T)

    ! This subroutine performs one Gauss-Seidel iteration
    ! over the matrix T to solve the heat equation
    ! n - matrix size
    ! T - n x n matrix of temperatures (including boundary terms)

    ! INPUT-OUTPUT VARIABLES
    integer, intent(in) :: n
    real*8, intent(inout) :: T(n,n)

    ! EXECUTION
    do i = 2,n-1
        do j = 2,n-1
            T(i,j) = 0.25*(T(i+1,j) + T(i,j+1) + T(i-1,j) + T(i,j-1))
        end do
    end do

end subroutine gauss_seidel