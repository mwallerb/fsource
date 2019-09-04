      SUBROUTINE TEST_SUB( X,
     &                     Y,
C        END SUBROUTINE   ! annoying!
     &                     Z )
      IMPLICIT INTEGER (I-K)
      INTEGER*8 X, Y,                                                  Z$IGNORE

      INTEGER R, S, T
      COMMON /MY/ R, S, /OTHER/ T

      DO 10 I = 1, 30
  39     X = Y + Z
  10  ENDDO

      DO 20 I = 1, 20
         DO 20 J = 1, 10
            X = Y - J * Z
  20  CONTINUE

      END

      PROGRAM
c Have a comment in between?
c
     &  SOMETHING

      INTEGER*8 X,Y,Z               ! comment
      CALL TEST_SUB(X,Y,Z)
      END PROGRAM
