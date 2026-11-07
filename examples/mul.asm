; M[2] = M[0] * M[1]   (repeated addition, M[1] >= 0)
        LOAD  r0, r3, 0      ; r3 is 0 at start: r0 = a
        LOAD  r1, r3, 1      ; r1 = b (loop counter)
        LOADI r3, 1
loop:   JZ    r1, done
        ADD   r2, r2, r0     ; acc += a
        SUB   r1, r1, r3     ; b -= 1
        JMP   loop
done:   STORE r2, r1, 2      ; r1 == 0 here, so this writes M[2]
        HALT

.data 0  7 6
