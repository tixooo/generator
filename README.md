Сваляме дейтасета:

wget http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar

инсталираме зависимости:

pip install -r requirements.txt

Подготвяме данните:

python prepare_data.py

тренираме:

python dcgan.py

генериране на изображения:

python generate.py

--------------
Генерираните изображения могат да се намерят в папка - generated след процеса
Можем да променим EPOCHS на по-малко число (например 10), за да тестваме по бързо - В dcgan.py EPOCHS=10 - промени от 25 на 10


за повече информация можем да видим training log.txt от единия от тестовете
