import numpy as np
#Phan 1:
# B1: PRINT all characters in “your name” by using for loops.

print("\nBai 1:")
name = "NguyenPhongPhu"
for i in name:
    print(i)

# B2: Print all odd numbers x such that 1<=x<=10

print("\nBai 2:")
for i in range(1, 11, 2):
    print(i)

# B3: a/Compute the sum of all numbers in 2/
# b/ Compute the sum of all number from 1 to 6.

print("\nBai 3:")
print("A.")
result = 0
for i in range(1, 11, 2):
    result = result + i
print("Tong cac so cua bai 2:", result)
print("B.")
result = 0
for i in range(0, 7):
    result = result + i
print("Tong cac so tu 1 toi 6", result)

# B4: Given mydict={“a”: 1,”b”:2,”c”:3,”d”:4}.
# a/ Print all key in mydict
# b/ Print all values in mydict
# c/ Print all keys and values

print("\nBai 4:")
mydict = {"a": 1, "b": 2, "c": 3, "d": 4}
print("A.")
print(mydict.keys())
print("B.")
print(mydict.values())
print("C.")
print(mydict.items())

# B5: Given courses=[131,141,142,212] and names=[“Maths”,”Physics”,”Chem”, “Bio”].
# Print a sequence of tuples, each of them contains one courses and one names

print("\nBai 5:")
courses = [131, 141, 142, 212]
names = ["Maths", "Physics", "Chem", "Bio"]
course_name = list(zip(courses, names))
print(course_name)

# B6: Find the number of consonants in “jabbawocky” by two ways
# a/ Directly (i.e without using the command “continue”)
# b/ Check whether it’s characters are in vowels set and using the command “continue”

print("\nBai 6:")
vowels = ["a", "o", "e", "u", "i"]
words = "jabbawocky"
print("A.")
count = sum(1 for char in words if char not in vowels)
print("So phu am: ", count)
print("B.")
count = 0
for char in words:
    if char in vowels:
        continue
    count += 1
print("So phu am: ", count)

#B7: a is a number such that -2<=a<3. Print out all the results of 10/a using try…except. When a=0, print out “can’t divided by zero”

print("\nBai 7:")
numbers = range(-2, 3, 1)
for i in numbers:
    try:
        print(f"10/{i}={10/i}")
    except ZeroDivisionError:
        print("can’t divided by zero")

#B8: Given ages=[23,10,80]
#And names=[Hoa,Lam,Nam]. Using lambda function to sort a list containing tuples (“age”,”name”) by increasing of the ages

print("\nBai 8:")
ages = [23,10,80]
names=["Hoa","Lam","Nam"]
people = list(zip(ages,names))
print("Danh sach sap xep theo tuoi: ", sorted(people, key = lambda  x : x[0]))

#B9: Create  a file “firstname.txt”:
# a/ Open this file for reading
# b/Print each line of this file
# c/ Using .read to read the file and
# Print it.

print("\nBai 9:")
with open("firstname.txt", "r") as file:
    print("B.")
    for line in file:
        print(line.strip())
    print("C.")
    file.seek(0)
    print(file.read())

#Phan 2:
#B1: Define a function that return the sum of two numbers a and b. Try with a=3, b=4.
print("\nBai 1:")
def sum_numbers(a,b):
    return a+b

print("Try with a=3, b=4")
print("Ket qua: ", sum_numbers(3,4))

#B2:  Create a 3x3 matrix M=■8(1&2&3@4&5&6@7&8&9) and vector v=■8(1&2&3)
# And check the rank and the shape of this matrix and vector v.
print("\nBai 2: ")
M = np.array([[1,2,3],[4,5,6],[7,8,9]])
rank = np.linalg.matrix_rank(M)
shape = M.shape
print("Ma tran: \n",M)
print("Hang: ", rank)
print("Kich thuoc: ",shape)

#B3: Create a new 3x3 matrix such that its’ elements are the sum of corresponding (position) element of M plus 3.
print("\nBai 3:")
new_matrix = M + 3
print("Ma tran moi: \n", new_matrix)

#B4: Create the transpose of M and v
print("\nBai 4:")
v = np.array([1, 2, 3])
M_transpose = M.T
v_transpose = v.T
print("Chuyển vị của ma trận M: \n", M_transpose)
print("Chuyển vị của vector v: \n", v_transpose)

#B5: Compute the norm of x=(2,7). Normalization vector x.
print("\nBai 5:")
x = np.array([2, 7])
norm_x = np.linalg.norm(x)
print("Norm of x:", norm_x)
x_normalized = x / norm_x
print("Normalized vector x:", x_normalized)

#B6: Given a=[10,15], b=[8,2] and c=[1,2,3]. Compute a+b, a-b, a-c. Do all of them work? Why?
print("\nBai 6:")
a = np.array([10, 15])
b = np.array([8, 2])
c = np.array([1, 2, 3])

sum_ab = a + b
print("a + b:", sum_ab)

diff_ab = a - b
print("a - b:", diff_ab)

try:
    diff_ac = a - c
    print("a - c:", diff_ac)
except ValueError as e:
    print("Error in a - c:", e)

#B7: Compute the dot product of a and b.
print("\nBai 7:")
dot_product = np.dot(a, b)
print("Dot product of a and b:", dot_product)

#Bai 8: Given matrix A=[[2,4,9],[3,6,7]].
	# a/ Check the rank and shape of A
	# b/ How can get the value 7 in A?
	# c/ Return the second column of A.
print("\nBai 8:")
print("A.")
A = np.array([[2, 4, 9], [3, 6, 7]])

shape_A = A.shape
rank_A = np.linalg.matrix_rank(A)

print("Shape of A:", shape_A)
print("Rank of A:", rank_A)

print("B.")
value_7 = A[1, 2]
print("Value 7 in A:", value_7)

print("C.")
second_column = A[:, 1]
print("Second column of A:", second_column)


#B9: Create a random  3x3 matrix  with the value in range (-10,10).
print("Bai 9:")
random_matrix = np.random.randint(-10, 10, size=(3, 3))
print("Random 3x3 matrix with values in range (-10, 10):\n", random_matrix)

#B10: Create an identity (3x3) matrix.
print("\nBai 10:")
identity_matrix = np.eye(3)
print("3x3 Identity Matrix:\n", identity_matrix)

#B11: Create a 3x3 random matrix with the value in range (1,10). Compute the trace of this matrix by 2 ways:
	# a/ By one command
	# b/ By using for loops
print("\nBai 11:")
random_matrix_2 = np.random.randint(1, 10, size=(3, 3))
print("Random 3x3 matrix with values in range (1, 10):\n", random_matrix_2)

# a/ Compute trace by one command
print("A.")
trace_1 = np.trace(random_matrix_2)
print("Trace of the matrix by one command:", trace_1)

# b/ Compute trace using for loops
print("B.")
trace_2 = 0
for i in range(3):
    trace_2 += random_matrix_2[i, i]
print("Trace of the matrix by for loops:", trace_2)

# B12: Create a 3x3 diagonal matrix with the value in main diagonal 1,2,3
print("\nBai 12:")
diagonal_matrix = np.diag([1, 2, 3])
print("3x3 Diagonal Matrix with values 1, 2, 3 in the main diagonal:\n", diagonal_matrix)

#B13: Given A=[[1,1,2],[2,4,-3],[3,6,-5]]. Compute the determinant of A
print("\nBai 13:")
A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
det_A = np.linalg.det(A)
print("Determinant of matrix A:", det_A)

#B14: Given a1=[1,-2,-5] and a2=[2,5,6]. Create a matrix M such that the first column is a1 and the second column is a2.
print("\nBai 14:")
a1 = np.array([1, -2, -5])
a2 = np.array([2, 5, 6])

M = np.column_stack((a1, a2))
print("Matrix M:\n", M)

#B15: Simply plot the value of the square of y with y in range (-5<=y<6).
print("\nBai 15:")
import matplotlib.pyplot as plt

y = np.arange(-5, 6)
y_square = y ** 2

plt.plot(y, y_square)
plt.title("Square of y")
plt.xlabel("y")
plt.ylabel("y^2")
plt.grid(True)
plt.show()

#B16: Create 4-evenly-spaced values between 0 and 32 (including endpoints)
print("\nBai 16:")
values = np.linspace(0, 32, 4)
print("4 evenly spaced values:", values)

#B17: Get 50 evenly-spaced values from -5 to 5 for x. Calculate y=x**2. Plot (x,y).
print("\nBai 17:")
x = np.linspace(-5, 5, 50)
y = x ** 2

plt.plot(x, y)
plt.title("Plot of y = x^2")
plt.xlabel("x")
plt.ylabel("y = x^2")
plt.grid(True)
plt.show()

#B18: Plot y=exp(x) with label and title.
print("\nBai 18:")
x = np.linspace(-2, 2, 100)
y = np.exp(x)

plt.plot(x, y)
plt.title("Plot of y = exp(x)")
plt.xlabel("x")
plt.ylabel("y = exp(x)")
plt.grid(True)
plt.show()

#B19: Similarly for y=log(x) with x from 0 to 5.
print("\nBai 19:")
x = np.linspace(0.1, 5, 100)
y = np.log(x)

plt.plot(x, y)
plt.title("Plot of y = log(x)")
plt.xlabel("x")
plt.ylabel("y = log(x)")
plt.grid(True)
plt.show()

#B20: Draw two graphs y=exp(x), y=exp(2*x) in the same graph and y=log(x) and y=log(2*x) in the same graph using subplot.
print("\nBai 20:")
x = np.linspace(-2, 2, 100)

fig, axs = plt.subplots(2, 1)

y1 = np.exp(x)
y2 = np.exp(2 * x)
axs[0].plot(x, y1, label="y = exp(x)")
axs[0].plot(x, y2, label="y = exp(2x)")
axs[0].set_title("y = exp(x) and y = exp(2x)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].legend()

x_log = np.linspace(0.1, 5, 100)
y_log1 = np.log(x_log)
y_log2 = np.log(2 * x_log)
axs[1].plot(x_log, y_log1, label="y = log(x)")
axs[1].plot(x_log, y_log2, label="y = log(2x)")
axs[1].set_title("y = log(x) and y = log(2x)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].legend()

plt.tight_layout()
plt.show()




