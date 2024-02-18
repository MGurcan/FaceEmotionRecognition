This project contains some Computer Vision Approaches into Facial Emotion Recognition

## Yöntem - 1:
Görüntü İşleme ve Feature Matching İçeren Yöntem
Bu yöntem herhangi bir yapay zeka desteği kullanılmadan PCA yardımı ile önemli olan
özniteliklerin(eigenfaces) tespiti ve öklid mesafesi ile tahmin yürütme üzerine olacaktır.
Yöntemin Adımları
➢ Ön İşleme
fer-2013 dataseti yüz imageleri kırpılmış ve 48x48 pixel hale getirilmiş şekilde
dataya sahiptir. Eğer farklı bir dataseti kullanılacak olursa yüz bölgesinin tespiti
yapılacak ve devamında normalize edilecektir.
➢ PCA Uygulama - EigenFaces Elde Etme
Gerekli olması halinde preprocess edilmiş datasetine PCA uygulanacak ve
verilerdeki en fazla varyansı yakalayan öznitelikler(eigenfaces) çıkartılacaktır. Bu
sayede görüntülerin boyutları azaltılacak, ancak önemli özelliklerin korunması
sağlanacaktır.
➢ Sınıflandırma - Öklid Mesafesi
Çıkarılan eigenfaces, test görüntülerinin duygu sınıfını belirlemek için kullanılacak,
her test görüntüsü için eigenface aracılığı ile elde edilen öznitelik vektörü eğitim
setindeki duygu sınıflarının ortalamaları ile öklid mesafesi kullanılarak
karşılaştırılacaktır. Karşılaştırma sonucunda elde edilen en kısa öklid mesafesi ise
test görüntüsünün ait olduğu duygu sınıfını belirleyecektir.

Performans Tahmini ve Değerlendirme
Hesaplama maliyetinin diğer yöntemlere göre daha düşük olması bir avantaj olarak
sayılabilir, ancak veri setindeki lineer olmayan ilişkileri yakalamak konusunda çok başarılı
olamayacağını düşündüğümüz bu yöntemde doğruluk oranının düşük olması gibi bir
dezavantaja sahip olabileceğimizi düşünüyoruz.

Kullanılan Makale ve Linkler
● A Human Facial Expression Recognition Model based on Eigen Face
Approach
(https://www.researchgate.net/publication/274096297_A_Human_Facial_Expre
ssion_Recognition_Model_Based_on_Eigen_Face_Approach)
● https://machinelearningmastery.com/face-recognition-using-principal-compone
nt-analysis/

## Yöntem - 2:
Gelişmiş Gabor Filtreleme, Öznitelik Eşleştirme ve Makine Öğrenmesi
Bu yöntem, yüzdeki geometrik özellikleri dikkate alarak farklı ifadeleri ayırt etme üzerine
geliştirilecektir. Gabor filtre, belirli yönlerde görüntü üzerinde herhangi bir spesifik frekans
içeriğinin olup olmadığını analiz etmede kullanılır. Gabor Filtre aracılığı ile üretilen
öznitelikler ile SVM, RF, knn gibi makine öğrenmesi yöntemleri beslenilir ve sınıflandırma
işlemi yapılır.
Yöntemin Adımları
➢ Ön İşleme:
fer-2013 dataseti yüz imageleri kırpılmış ve 48x48 pixel hale getirilmiş şekilde
dataya sahiptir. Eğer farklı bir dataseti kullanılacak olursa yüz bölgesinin tespiti
yapılacak ve devamında normalize edilecektir.
➢ Gabor Filtre:
Çeşitli parametreler ile gabor filtreleme işlemleri uygulanacaktır. Burada
uygulanacak filtre parametreleri aşağıdaki denklemdeki parametreler olacaktır.

def gabor(image, frequency, theta=0, bandwidth=1, sigma_x=None,
sigma_y=None, n_stds=3, offset=0, mode='reflect', cval=0):
Yukarıda signature’u verilen simage.filters - gabor fonksiyonu kullanılacak ve
farklı parametreler denenecektir. Bu denemeler yapılırken üretilen çıktının
f1-score accuracy gibi değerleri kullanılacaktır. Ayrıca denemeler aşağıda bahsi
geçen makaledeki değerler ve çevresini de dikkate alacaktır. (Orientation değeri
-45, 90, 45, 0 ; Aspect Ratio 0.3 gibi)
➢ Feature Extraction - Matching
Bu aşama için patch-based feature extraction kullanılması planlanmaktadır. Farklı
boyuttaki patch’ler ‘Gabor Filtresi’ uygulanan yüz bölgesinden çıkartılacaktır.
Çıkartılan patch’ler belirlenen eşleştirme alanları içerisinde diğer patch’ler ile
karşılaştırılarak özniteliklere dönüştürülecektir.
➢ Salient Patch Seçimi
Adaboost algoritması yardımı ile çıkarılan öznitelikler arasında daha bilgilendirici
olan patch’ler(salient patch) seçilecektir.
➢ Feature Matching - Classification
Bu işlemler sonucunda ortay çıkan öznitelikler yüz ifadesi label’ları ile eşleştirilecek
ve ‘SVM’, ‘Random Forest’, ‘kNN’ kullanılarak sınıflandırılacaktır.
Bu aşama için farklı modellerin parametreleri de denenecektir. Random Forest için
n_estimator(ağaç sayısı), max_depth gibi parametreler üzerine yoğunlaşılacaktır.
Model başarımlarına göre bu parametrelere yenileri eklenecektir. SVM için kernel
ve gamma başta olmak üzere parametreler seçilecektir. kNN için ise n_neighbors
gibi parametreler üzerine yoğunlaşılacaktır.
Parametrelerin optimizasyonu sırasında GridSearchCV kullanılması da
planlanmaktadır.

Performans Tahmini ve Değerlendirme
CNN yöntemine göre daha düşük doğruluk oranına sahip olacağı düşünülmektedir, ancak
yine de Gabor Filtrelerin yüzün detaylı karakteristiklerini yakalama kapasitesi sayesinde iyi
bir oran olacağı tahmin edimektedir. Farklı parametrelerin denenmesi ve patch-based
yaklaşım ile bu oranda bir artış olacağ da düşünülmektedir.

Farklı parametrelere göre filtrelerin denenmesi ve daha önce bu alanda bir çalışma
yapılmamasından dolayı parametrelerin etkisinin tam kestirilememesi, ve buna bağlı olarak
uzun bir işleme süresine sahip olabileceği dezavantajını düşünüyoruz. Doğru
parametrelerin yakalanması durumunda ise Gabor Filtrelerin yüzün detaylı karakteristiğini
yakalayabileceğini ve salient features seçimleri ile duyguyu belirleyen önemli
özniteliklerin çıkarılabileceği ile birlikte iyi bir tahmin yüzdesi yakalayabileceği avantajına
sahip olmasını bekliyoruz.

Kullanılan Makale ve Linkler:
● Facial Expression Recognition Using Facial Movement Features
(https://ieeexplore.ieee.org/document/5871583)
● https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf
(“from skimage.filters import gabor” fonksiyonu referans göstermiştir)
● https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac9
7
● https://www.baeldung.com/cs/ml-gabor-filters

## Yöntem 3
CNN İçeren Yöntem
Bu aşamada feature extraction gibi işlemler uygulanmayacak, vanilla CNN üzerine çeşitli
parametreler denenecek, ResNet50, SeNet50, VGG16 gibi pre-trained modeller fine tune
edilerek yüksek başarımlar elde edilmeye çalışılacaktır.

Yöntemin Adımları
1. Vanilla CNN
Herhangi bir pre-trained kullanılmadan denenecek olan bu CNN aşamasında farklı
aktivasyon fonksiyonları, katmanlar, batch normalizasyonları ve dropout denemeleri
ile başarımlar ölçülecek, buna bağlı olarak parametre optimizasyonları denenecektir.
“Facial Expression Recognition with Deep Learning” çalışmasındaki parametreler
başlangıç değerleri için denenip üzerinde oynama yapılması beklenmektedir. Bunları tekrar
sayacak olursak; ilk denemelerimizin RELU aktivasyon fonksiyonu, MaxPool katmanı,
FC katmanı, softmax katmanı içermesini ve farklı dropout değerlerinin de denenmesini
planlamaktayız.
“FER-2013” datasetinin yeterince büyük olmaması sebebi ile pre-trained model
kullanılmadan sıfırdan eğitilen CNN’in accuracy beklentisini 60%-65% civarında
tutuyoruz.

2. Fine Tuning Pre-Trained Models

Projenin bu kısmında önceden belirlemiş olduğumuz 3 farklı pre-trained model’e fine tune
denemeleri yapacağız. Bunun için Keras VGG-Face kütüphanesini kullanarak ResNet50,
SeNet50, VGG16 pre-trained modellerini denemeyi planlıyoruz.
Yapmış olduğumuz literatür taramasında bu modellerin 197x197’den küçük olmayan RGB
image’ler istediği yönünde bilgiler topladık. Bunun için dataset üzerine modellere uygun
hale getirme amacıyla çeşitli preprocess işlemleri yaparak çalışmamıza başlayacağız
(Resizing-Coloring).
ResNet50, 50 katmandan oluşan bir derin kalıntı ağıdır. Keras içerisinde 175 katman olarak
yer almaktadır.
ResNet50 modeline fine tune işlemi için ilk değişiklik 4096 ve 1024 boyutlu 2 FC layer ve
7 duygu sınıfı çıkışına sahip softmax layer kullanmayı planlıyoruz. Elde ettiğimiz başarıma
göre bu parametreler üzerine değişiklikler yapacağız. Optimizer olarak SGD ilk aşamada
kullanılacak , learning rate 0.01 olarak alınacak ve ilk denemede batch size 32 olarak
kullanılacaktır. İleriki süreçte bu parametreler üzerine farklı denemeler de yapılacaktır.
Literatürde ResNet50 modeli FER-2013 dataseti ile fine tune edilme örneklerine sahiptir ve
accuracy değerleri 70%-75% arasında değişmektedir. Bizim beklentimiz de bu bantta yer
almaktadır.
SeNet50, 50 katmanlı başka bir derin kalıntı ağıdır. ResNet50’ye benzer bir pre-trained
model olduğu için benzer parametreler iki model üzerinde denenecek ve accuracy değerleri
incelenecektir.
Bu model için de accuracy beklentimiz ResNet50’ye benzer olarak 70%-75% bandındadır.
VGG16, diğer iki modele kıyasla daha sığ bir modeldir ve 16 katmandan oluşur. Daha az
katmana sahip olduğu için tüm katmanlarının tutulması ve üzerine ek katman ekleme
denemelerini yapmayı planlamaktayız. Farklı boyutlarda FC layerlar eklenecektir.
Accuracy değerimizi diğer iki modele kıyasla daha sığ model olduğu için 65%-70%
bandında düşünüyoruz.

Kullanılan Makale ve Linkler:
Facial Expression Recognition with Deep Learning
(https://arxiv.org/abs/2004.11823)