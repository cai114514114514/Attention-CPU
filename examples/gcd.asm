; M[2] = gcd(M[0], M[1])   (subtraction Euclid, positive inputs)
        LOAD  r0, r3, 0      ; a   (r3 stays 0 and serves as base register)
        LOAD  r1, r3, 1      ; b
loop:   SUB   r2, r0, r1
        JZ    r2, done
        JNEG  r2, bless
        SUB   r0, r0, r1     ; a > b: a -= b
        JMP   loop
bless:  SUB   r1, r1, r0     ; a < b: b -= a
        JMP   loop
done:   STORE r0, r3, 2
        HALT

.data 0  1071 462
