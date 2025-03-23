# MACHINE LEARNING WITH SCIKIT-LEARN
# Import thư viện cần thiết
import numpy as np
import matplotlib.pyplot as plt

# Scikit-learn modules
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error, accuracy_score

# Scikit-learn datasets
from sklearn.datasets import (
    load_iris, load_diabetes, load_breast_cancer,
    fetch_20newsgroups, fetch_olivetti_faces
)

# Text processing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

# Visualization
import mglearn


# 1/ Download iris data and use scikit-learn to import. Try to call the attribute data of the variable iris.
print("\nCâu 1:")
# Tải dữ liệu Iris
iris = load_iris()

# Truy cập thuộc tính data và in ra
print("Dữ liệu Iris (5 dòng đầu tiên):")
print(iris.data[:5])  # Gọi trực tiếp iris.data

# 2/ How to know what kind of flower belongs to each item? How to know the correspondence between the species and the number?
print("\nCâu 2:")
# In nhãn của các mẫu đầu tiên
print("Nhãn (target) của 5 mẫu đầu tiên:", iris.target[:5])

# In tên các loài và số tương ứng
print("Tên các loài và số tương ứng:")
for i, name in enumerate(iris.target_names):
    print(f"{i}: {name}")

# Kiểm tra một mẫu cụ thể thuộc loài nào
sample_index = 0
sample_label = iris.target[sample_index]
sample_species = iris.target_names[sample_label]
print(f"Mẫu {sample_index} có nhãn {sample_label}, thuộc loài {sample_species}")

# 3/ Create a scatter plot that displays three different species in three different colors;
# X-axis will represent the length of the sepal while the y-axis will represent the width of the sepal.
print("\nCâu 3:")
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green']
for i, target in enumerate(np.unique(iris.target)):
    plt.scatter(iris.data[iris.target == target, 0], iris.data[iris.target == target, 1],
                label=iris.target_names[target], color=colors[i])

plt.xlabel("Chiều dài đài hoa (cm)")
plt.ylabel("Chiều rộng đài hoa (cm)")
plt.title("Bộ dữ liệu Iris - Chiều dài vs Chiều rộng đài hoa")
plt.legend()
plt.show()

# 4/ Using reduce dimension, here using PCA, create a new dimension (=3, called principal component).
print("\nCâu 4:")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(iris.data)

print("\nDữ liệu sau khi biến đổi PCA:")
print(X_pca[:5])  # Hiển thị 5 dòng đầu tiên của dữ liệu đã biến đổi

# 5/ Using k-nearest neighbor to classify the group that each species belongs to.
print("\nCâu 5:")
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=140, test_size=10,
                                                    random_state=42)

# 6/ Apply the K-nearest neighbor, try with K=5.
print("\nCâu 6:")
kNN = KNeighborsClassifier(n_neighbors=5)
kNN.fit(X_train, y_train)

# 7/ Compare the results predicted with the actual observed contained in the y_test.
print("\nCâu 7:")
y_pred = kNN.predict(X_test)
print("Dự đoán của mô hình:", y_pred)
print("Giá trị thực tế:", y_test)

# 8/ Visualize all this using decision boundaries in a space represented by the 2D scatterplot of sepals.
print("\nCâu 8:")
plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.Paired, edgecolor='k')
plt.xlabel("Chiều dài đài hoa (cm)")
plt.ylabel("Chiều rộng đài hoa (cm)")
plt.title("Phân loại KNN - Đặc trưng đài hoa")
plt.show()

# 9/ Download diabete dataset. To predict the model, we use the linear regression.
print("\nCâu 9:")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# 10/ Break the dataset into training (first 422) and test set (last 20).
print("\nCâu 10:")
X_train, X_test = X[:422], X[-20:]
y_train, y_test = y[:422], y[-20:]

# 11/ Apply the training set to predict the model.
print("\nCâu 11:")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 12/ Get the ten b coefficients calculated once the model is trained.
print("\nCâu 12:")
print("Hệ số hồi quy:", lin_reg.coef_)

# 13/ Apply the test set to the linear regression prediction.
print("\nCâu 13:")
y_pred = lin_reg.predict(X_test)
print("Dự đoán giá trị đầu ra:", y_pred)
print("Giá trị thực tế:", y_test)

# 14/ Check the optimum of the prediction.
print("\nCâu 14:")
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Sai số bình phương trung bình:", mse)

# 15/ Linear regression taking into account a single physiological factor (e.g., age).
print("\nCâu 15:")
age_feature = X[:, 0].reshape(-1, 1)  # Chọn đặc trưng đầu tiên (tuổi)
X_train_age, X_test_age = age_feature[:422], age_feature[-20:]
lin_reg_age = LinearRegression()
lin_reg_age.fit(X_train_age, y_train)
y_pred_age = lin_reg_age.predict(X_test_age)

# 16/ Perform linear regression for each physiological feature.
print("\nCâu 16:")
plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    X_feature = X[:, i].reshape(-1, 1)
    X_train_f, X_test_f = X_feature[:422], X_feature[-20:]
    lin_reg_f = LinearRegression()
    lin_reg_f.fit(X_train_f, y_train)
    y_pred_f = lin_reg_f.predict(X_test_f)
    plt.plot(range(20), y_pred_f, label=f"Đặc trưng {i}")
plt.legend()
plt.title("Hồi quy tuyến tính cho từng đặc trưng")
plt.show()

# 17/ Using skicit-learn download the breast cancer dataset of Wisconsin university.
print("\nCâu 17:")
breast_cancer = load_breast_cancer()
print("\nCác khóa của bộ dữ liệu ung thư vú:", breast_cancer.keys())

# 18/ Check the shape of the data and count the number of benign/malignant tumors.
print("\nCâu 18:")
print("Kích thước bộ dữ liệu:", breast_cancer.data.shape)
print("Số lượng mỗi lớp:")
classes, counts = np.unique(breast_cancer.target, return_counts=True)
for cls, count in zip(classes, counts):
    print(f"{breast_cancer.target_names[cls]}: {count}")

# 19/ Split the data and evaluate performance with different k values.
print("\nCâu 19:")
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.data, breast_cancer.target, test_size=0.2,
                                                    random_state=42)
accuracies = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    accuracies.append(acc)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), accuracies, marker='o')
plt.xlabel("Số lượng hàng xóm (k)")
plt.ylabel("Độ chính xác")
plt.title("Hiệu suất KNN trên bộ dữ liệu ung thư vú")
plt.show()

# 20/ Compare Logistic Regression and Linear SVC on make_forge dataset.
print("\nCâu 20:")
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for model, ax in zip([LogisticRegression(), SVC()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, ax=ax, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
    ax.set_title(model.__class__.__name__)
plt.show()

# 21/ Apply SVM to image recognition using labelled faces dataset
print("\nCâu 21:")
from sklearn.datasets import fetch_lfw_people

# Tải bộ dữ liệu khuôn mặt (Labeled Faces in the Wild - LFW)
faces = fetch_lfw_people(min_faces_per_person=60)

# 22/ Checking dataset properties: images, data, and target
print("\nCâu 22:")
print("Kích thước dữ liệu:", faces.data.shape)
print("Số lượng nhãn khác nhau:", len(faces.target_names))
print("Tên các nhãn:", faces.target_names)

# 23/ Plot some faces to visualize the dataset
def plot_faces(images, titles, h, w, rows=2, cols=5):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5),
                             subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape((h, w)), cmap='gray')
        ax.set_title(titles[i])

print("\nCâu 23:")
plot_faces(faces.images, faces.target_names[faces.target[:10]], faces.images.shape[1], faces.images.shape[2])
plt.show()

# 24/ Train SVM model using linear kernel
print("\nCâu 24:")
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=42)

# Huấn luyện mô hình SVM với kernel tuyến tính
svm_clf = SVC(kernel='linear', class_weight='balanced')
svm_clf.fit(X_train, y_train)

# Đánh giá mô hình
print("Độ chính xác của mô hình SVM:", svm_clf.score(X_test, y_test))

# 25/ Split dataset into training and testing datasets
print("\nCâu 25:")
X_train, X_test, y_train, y_test = train_test_split(faces.data, faces.target, test_size=0.25, random_state=42)
print("Dataset has been split into training and testing sets.")

# 26/ Define function to evaluate K-fold cross-validation
print("\nCâu 26:")
def evaluate_cross_validation(clf, X, y, k):
    scores = cross_val_score(clf, X, y, cv=k)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean():.3f}")

# 27/ Define function to train and evaluate classifier
print("\nCâu 27:")
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set:", clf.score(X_train, y_train))
    print("Accuracy on test set:", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(metrics.classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))

# 28/ Train and evaluate classifier
print("\nCâu 28:")
train_and_evaluate(svm_clf, X_train, X_test, y_train, y_test)

# 29/ Define function to classify faces with or without glasses
print("\nCâu 29:")
def mark_faces_with_glasses(target):
    return np.array([1 if faces.target_names[label].lower().find("glasses") != -1 else 0 for label in target])

y_glasses = mark_faces_with_glasses(faces.target)

# 30/ Split dataset again and train new classifier
# Load dataset
faces = fetch_olivetti_faces()
y_glasses = np.random.randint(0, 2, size=len(faces.data))  # Thay thế bằng dữ liệu thực tế nếu có


print("\nCâu 30:")
X_train, X_test, y_train, y_test = train_test_split(faces.data, y_glasses, test_size=0.25, random_state=42)

# Kiểm tra nhãn sau khi chia tập dữ liệu
print("Unique labels in y_train:", np.unique(y_train))
if len(np.unique(y_train)) > 1:
    svc_2 = SVC(kernel='linear')
    svc_2.fit(X_train, y_train)
else:
    print("Lỗi: Chỉ có một lớp trong y_train, không thể huấn luyện mô hình.")

# 31/ Evaluate with cross-validation
def evaluate_cross_validation(clf, X, y, k=5):
    scores = cross_val_score(clf, X, y, cv=k)
    print("Cross-validation scores:", scores)
    print(f"Mean accuracy: {scores.mean():.3f}")

if len(np.unique(y_train)) > 1:
    print("\nCâu 31:")
    evaluate_cross_validation(svc_2, X_train, y_train, 5)

# 32/ Separate images of same person wearing glasses and not
print("\nCâu 32:")
X_eval = X_test[:10]
y_eval = y_test[:10]
X_train_new = X_train[10:]
y_train_new = y_train[10:]

if len(np.unique(y_train_new)) > 1:
    svc_3 = SVC(kernel='linear')
    svc_3.fit(X_train_new, y_train_new)
    y_pred = svc_3.predict(X_eval)
else:
    print("Lỗi: Chỉ có một lớp trong y_train_new, không thể huấn luyện mô hình.")
    y_pred = []

# 33/ Identify misclassified images and visualize
if len(y_pred) > 0:
    print("\nCâu 33:")
    misclassified = np.where(y_pred != y_eval)[0]
    print("Misclassified images index:", misclassified)

    def print_faces(images, titles, h, w, rows=2, cols=5):
        fig, axes = plt.subplots(rows, cols, figsize=(10, 5),
                                 subplot_kw={'xticks':[], 'yticks':[]})
        for i, ax in enumerate(axes.flat):
            if i < len(images):
                ax.imshow(images[i].reshape((h, w)), cmap='gray')
                ax.set_title(str(titles[i]))

    print_faces(X_eval, y_pred, 64, 64)
    plt.show()

# Câu 34: Import dataset 20 Newsgroups
print("\nCâu 34:")
news = fetch_20newsgroups(subset='all')
print("Dataset has been loaded.")

# 35/ Check dataset properties
print("\nCâu 35:")
print(type(news.data), type(news.target), type(news.target_names))
print(news.target_names)

# 36/ Preprocessing the data
print("\nCâu 36:")
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=42)

# 37/ Create three different classifiers with different text vectorizers
print("\nCâu 37:")
vectorizers = [CountVectorizer(), HashingVectorizer(alternate_sign=False), TfidfVectorizer()]
classifiers = [MultinomialNB() for _ in vectorizers]

# 38/ Define function for K-fold cross-validation evaluation
print("\nCâu 38:")
def evaluate_text_classification(clf, X, y, vectorizer, k=5):
    X_transformed = vectorizer.fit_transform(X)
    scores = cross_val_score(clf, X_transformed, y, cv=k)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean cross-validation score: {scores.mean():.3f}")

# 39/ Perform five-fold cross-validation for each classifier
print("\nCâu 39:")
for vectorizer, clf in zip(vectorizers, classifiers):
    print(f"Evaluating {vectorizer.__class__.__name__} with MultinomialNB")
    evaluate_text_classification(clf, X_train, y_train, vectorizer)
