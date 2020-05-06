module test
    ! Check whether implicits are handled correctly
    implicit integer (a-r)
    implicit character(3) (t-u)

    ! Check if string concatenation is handled properly
    character(len=*), parameter :: x = 'a string that contains &
        this extension and also &
        end module !&
        and ends here' // 'next'

    ! Check if multi-line macros are handled properly
    ! (also, annoy regex tools)
#define MY_END_SUBROUTINE \
    END SUBROUTINE ! \
    PROGRAM x

    ! Check whether in-between line-continuated
    integer, parameter :: y = 1 &
            ! here's a comment
                        + 2 &
            ! here's another one
                        + 3

    double complex :: A(10) = (/ (1.0,4.0), (CMPLX(1.0*I), I=1,8), (3.0,10.0) /)

    type :: mytype
        real :: x
    end type

    ! Check annoying ambiguity between in-place arrays and operators
    interface operator( /)
      module procedure divideme
    end interface operator(/ )

    interface operator( / )
      module procedure divideyou
    end interface operator(/)

    interface operator( //)
      module procedure divideme
    end interface operator(// )

    interface operator( // )
      module procedure divideyou
    end interface operator(//)


contains

    real function divideme(x, y)
        type(mytype), intent(in) :: x, y
        divideme = x%x + y%x
    endfunction

    real function divideyou(x, y)
        type(mytype), intent(in) :: x
        real, intent(in) :: y
        divideyou = x%x + y
    end

    subroutine nastyconsts
        integer :: i, ge
        logical :: j

        j = ge.GE.3.AND. 4.ne.i
    end subroutine

    ! Annoy the regex tools
end &   ! subroutine
    module

program test_program
    ! Yes, this is legal
    integer :: endif, end

    if (endif == 1) then
        endif = 3
    endif
    end & ! program
        = 6
end
