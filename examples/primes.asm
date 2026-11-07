; OUT every prime p < 50: trial division by repeated subtraction, MUL for the d*d > p test.
        LOADI r0, 2          ; p
next_p: LOADI r1, 2          ; d
try_d:  MUL   r2, r1, r1     ; d*d
        SUB   r2, r0, r2     ; p - d*d
        JNEG  r2, prime      ; d*d > p  ->  p is prime
        MOV   r2, r0         ; r = p
mod:    SUB   r2, r2, r1     ; r -= d
        JZ    r2, composite  ; d divides p
        JNEG  r2, notdiv
        JMP   mod
notdiv: ADDI  r1, r1, 1
        JMP   try_d
prime:  OUT   r0
composite:
        ADDI  r0, r0, 1
        ADDI  r3, r0, -50
        JNEG  r3, next_p
        HALT
