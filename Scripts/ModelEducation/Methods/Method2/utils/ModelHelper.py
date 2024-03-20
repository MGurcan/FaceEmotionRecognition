import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from skimage.filters import gabor_kernel
from scipy.ndimage import convolve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.svm import SVC

def apply_hog_and_pca(X_train, X_test, y_train, y_test, shape=48):
    X_train_hog = extract_hog_features(X_train, shape)
    X_test_hog = extract_hog_features(X_test, shape)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_hog)
    X_test_pca = pca.transform(X_test_hog)
    return X_train_pca, X_test_pca

def apply_hog_and_lda(X, y):
    X_hog = extract_hog_features(X)
    lda = LDA(n_components=3)
    X_lda = lda.fit_transform(X_hog, y)
    return X_lda

def extract_hog_features(images, shape=48):
    hog_features = []
    for image in images:
        image_resized = image.reshape(shape, shape)
        fd = hog(image_resized, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)


def apply_gabor_filter(image, kernel):
    return convolve(image, kernel.real) + convolve(image, kernel.imag)

def extract_gabor_features(image, kernels):
    img_reshaped = image.reshape(48, 48)
    features = []
    for kernel in kernels:
        filtered = apply_gabor_filter(img_reshaped, kernel)
        features.append(filtered.mean())
        features.append(filtered.var())
    return features

def apply_gabor_and_pca(X, y):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                kernels.append(kernel)

    X_gabor = np.array([extract_gabor_features(image, kernels) for image in X])
    pca = PCA(n_components=0.95)  # %95 varyansı koruyacak şekilde boyut azaltma
    X_pca = pca.fit_transform(X_gabor)
    return X_pca

def apply_gabor_and_lda(X, y):
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma)
                kernels.append(kernel)

    X_gabor = np.array([extract_gabor_features(image, kernels) for image in X])
    lda = LDA()  # LDA nesnesi oluşturulur, varsayılan olarak tüm bileşenleri kullanır
    X_lda = lda.fit_transform(X_gabor, y)  # LDA, hem fit hem de transform işlemini yapar
    return X_lda

def classify_with_knn(X_train, X_test, y_train, y_test, n_neighbors=5):
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred_knn = knn_classifier.predict(X_test)

    # Tahmin başarımını değerlendirme
    accuracy = accuracy_score(y_test, y_pred_knn)
    print(f'Test seti üzerindeki doğruluk: {accuracy:.2f}')
    return y_pred_knn

def classify_with_svm(X_train, X_test, y_train, y_test):
    svm_classifier = SVC()

    svm_classifier.fit(X_train, y_train)

    y_pred_svm = svm_classifier.predict(X_test)

    # Tahmin başarımını değerlendirme
    accuracy = accuracy_score(y_test, y_pred_svm)
    print(f'Test seti üzerindeki doğruluk: {accuracy:.2f}')
    return y_pred_svm

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def predict_with_euclidean_distance(X_train, y_train, X_test):
    print("X_Test lenght: ", len(X_test))
    predictions = []
    counter = 0
    for test_sample in X_test:
        counter += 1
        if counter % 100 == 0:
            print("%", counter / len(X_test) * 100)
        distances = [euclidean_distance(
            test_sample, train_sample) for train_sample in X_train]
        min_index = np.argmin(distances)
        predictions.append(y_train[min_index])
    return predictions

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cosine_distance(a, b):
    # Kosinüs benzerliğini hesaplayıp bunu mesafe olarak döndürmek için 1'den çıkarırız.
    similarity = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
    return 1 - similarity

def predict_with_cosine_distance(X_train, y_train, X_test):
    print("X_Test length: ", len(X_test))
    predictions = []
    counter = 0
    for test_sample in X_test:
        counter += 1
        if counter % 100 == 0:
            print("%", counter / len(X_test) * 100)
        distances = [cosine_distance(test_sample, train_sample) for train_sample in X_train]
        min_index = np.argmin(distances)
        predictions.append(y_train[min_index])
    return predictions



default_params = {'n_components': 150, 'svd_solver': 'auto',
                          'iterated_power': 7, 'random_state': 42}


def sklearn_PCA(X_train, X_test, param=default_params):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pca = PCA(n_components=param['n_components'],
              svd_solver=param['svd_solver'],
              iterated_power=param['iterated_power'],
              random_state=param['random_state'])
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca


def apply_sklearn_PCA(X_train, X_test, y_train, param=default_params):
    X_train_pca, X_test_pca = sklearn_PCA(X_train, X_test, param)
    predictions = predict_with_euclidean_distance(
        X_train_pca, y_train, X_test_pca)
    return predictions


def manuel_PCA(X_train, X_test, variance_threshold):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mean = np.mean(X_train, axis=0)
    X_centered = X_train - mean

    cov_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[sorted_indices]
    eigenvectors_sorted = eigenvectors[:, sorted_indices]

    total_variance = sum(eigenvalues_sorted)
    cumulative_variance_ratio = np.cumsum(eigenvalues_sorted) / total_variance

    n_components = np.where(cumulative_variance_ratio >= variance_threshold)[0][0] + 1

    reduced_eigenvectors = eigenvectors_sorted[:, :n_components]

    X_train_pca = np.dot(X_centered, reduced_eigenvectors)
    X_test_pca = np.dot(X_test - mean, reduced_eigenvectors)
    return X_train_pca, X_test_pca

def apply_manuel_PCA(X_train, X_test, y_train, variance_threshold = 0.95):
    X_train_pca, X_test_pca = manuel_PCA(X_train, X_test, variance_threshold)
    predictions = predict_with_euclidean_distance(X_train_pca, y_train, X_test_pca)
    # predictions = predict_with_cosine_distance(X_train_pca, y_train, X_test_pca)
    
    return predictions