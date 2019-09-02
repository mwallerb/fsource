module simplemod
    use, intrinsic :: iso_c_binding, only: c_int
contains
    subroutine doubleval(x, y)
        integer(c_int), intent(in) :: x
        integer(c_int), intent(out) :: y

        y = 2 * x
    end subroutine
end module
