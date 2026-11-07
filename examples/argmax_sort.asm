; OUT the 8 numbers in M[0..7] from largest to smallest with the ARGMAX vector instruction:
; each round, one attention head scans all 64 memory words at once.
; Every other word is filled with the minimum value so it never wins.
        LOADI  r2, -32768    ; knock-out value
        LOADI  r3, 8         ; rounds left
loop:   ARGMAX r1            ; r1 = address of the largest word in memory
        LOAD   r0, r1, 0
        OUT    r0
        STORE  r2, r1, 0     ; remove it
        ADDI   r3, r3, -1
        JZ     r3, done
        JMP    loop
done:   HALT

.data 0  5 -3 12 0 7 7 -20 4
.fill 8 56 -32768
