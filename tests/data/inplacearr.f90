module inplacearr
    use iso_c_binding

    type :: testtype
        integer :: v
    end type

    ! Check whether we understand (//)
    interface operator (//)
        module procedure add_testtypes
    end interface operator (//)

    integer, parameter :: arr(4) = (/ 1, 2, (2*i, i=1, 2) /)

    real(c_double), parameter :: nan = &
                        transfer((/ Z'00000000', Z'7FF80000' /), 1.0_c_double)

contains
    function add_testtypes(a, b)
        type(testtype), intent(in) :: a, b
        type(testtype) :: add_testtypes

        add_testtypes%v = a%v + b%v
    end function
end module

program testprog
    use inplacearr
end program
