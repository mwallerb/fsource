module inplacearr
    type :: testtype
        integer :: v
    end type

    ! Check whether we understand (//)
    interface operator (//)
        module procedure add_testtypes
    end interface operator (//)

    integer, parameter :: arr(4) = (/ 1, 2, (2*i, i=1, 2) /)
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
