; M[0..19] = F(0) .. F(19)
; i runs from -20 up to 0 and is used as the store base (address i + 20).
        LOADI r0, 0          ; a = F(k)
        LOADI r1, 1          ; b = F(k+1)
        LOADI r2, -20        ; i
        LOADI r3, 1
loop:   STORE r0, r2, 20     ; M[k]   = a
        ADD   r0, r0, r1     ; a = F(k+2)
        ADD   r2, r2, r3
        STORE r1, r2, 20     ; M[k+1] = b
        ADD   r1, r1, r0     ; b = F(k+3)
        ADD   r2, r2, r3
        JNEG  r2, loop
        HALT
