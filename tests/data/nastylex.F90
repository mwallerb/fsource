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
