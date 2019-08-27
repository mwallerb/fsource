C0000&...:....,....:....,....:....,....:....,....:....,....:....,....:..
*     Here's a comment
! 0 Here's another one
      SUBROUTINE TEST_SUB( X,
     &                     Y,
C        END SUBROUTINE   ! annoying!
     &                     Z )
      IMPLICIT INTEGER (I-K)
      INTEGER*8 X, Y,                                                  Z$IGNORE

      INTEGER R, S, T
      COMMON /MY/ R, S, /OTHER/ T

      DO 10 I = 1, 10
  39      X = Y
     &        + Z
   10 CONTINUE
      END

      PROGRAM
c Have a comment in between?
c
     &  SOMETHING

      INTEGER*8 X,Y,Z               ! comment
      CALL TEST_SUB(X,Y,Z)
      END PROGRAM
