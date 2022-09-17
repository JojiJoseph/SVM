from linear_regression import LinearRegression

reg = LinearRegression()

reg.fit([[0, 0], [1, 1], [2, 2]], [1, 2, 3])

print(reg.w)  # About [0.5, 0.5]
print(reg.b)  # About 1
