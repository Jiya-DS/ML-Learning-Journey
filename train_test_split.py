from sklearn.model_selection import train_test_split

X=[1,5,4,3,8,6,0,9,7]
y=[23,14,35,26,57,90,89,67,30]

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25,random_state=23)
print("Training data:",X_train)
print("Testing data:", X_test)

# Sample data
# X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # features
# y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # target

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print("Training data:", X_train)
# print("Testing data:", X_test)
# print("Training size:", len(X_train))
# print("Testing size:", len(X_test))

# from sklearn.model_selection import train_test_split

# X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# # With random_state
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print("Fixed split:", X_test)

# # Without random_state
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print("Random split:", X_test)

