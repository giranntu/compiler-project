	.text
	.globl	safe_run
	.type	safe_run,@function
safe_run:           
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	pushq	%rbp
	pushq	%rax
	callq	run
        addq    $8, %rsp
	popq	%rbp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq

