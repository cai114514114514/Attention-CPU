; OUT n! for n = 1..7 with a recursive fact(), using CALL/RET, the stack and MUL.
        LOADI r1, 1          ; n
loop:   MOV   r0, r1         ; argument
        CALL  fact           ; r0 = fact(r0)
        OUT   r0
        ADDI  r1, r1, 1
        ADDI  r2, r1, -8
        JNEG  r2, loop
        HALT

; fact(r0) -> r0     (uses r2, r3; keeps r1)
fact:   ADDI  r2, r0, -1
        JNEG  r2, base       ; n <= 0  -> 1
        JZ    r2, base       ; n == 1  -> 1
        PUSH  r0             ; save n
        MOV   r0, r2
        CALL  fact           ; r0 = fact(n - 1)
        POP   r3             ; n
        MUL   r0, r0, r3     ; 17 clock steps
        RET
base:   LOADI r0, 1
        RET
