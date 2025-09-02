fibonacci = lambda n: 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)  # Recursive function to calculate Fibonacci numbers
fizz_buzz = [(i, "Fizz"*(i%3==0) + "Buzz"*(i%5==0) or i) for i in range(1, 101)]  # classic interview problem
