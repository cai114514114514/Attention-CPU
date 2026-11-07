; Bubble sort M[0..7] ascending. M[63] holds the outer pass counter (n - 1).
outer:  LOADI r0, 0          ; j = 0
inner:  LOAD  r1, r0, 0      ; x = M[j]
        LOAD  r2, r0, 1      ; y = M[j+1]
        SUB   r3, r2, r1     ; y - x
        JNEG  r3, swap       ; y < x  ->  swap
next:   LOADI r3, 1
        ADD   r0, r0, r3     ; j += 1
        LOADI r3, 7
        SUB   r3, r0, r3     ; j - 7
        JNEG  r3, inner
        LOAD  r1, r0, 56     ; j == 7 here, so this is M[63]
        LOADI r3, 1
        SUB   r1, r1, r3
        STORE r1, r0, 56
        JZ    r1, done
        JMP   outer
done:   HALT
swap:   STORE r2, r0, 0
        STORE r1, r0, 1
        JMP   next

.data 0  5 -3 12 0 7 7 -20 4
.data 63 7
